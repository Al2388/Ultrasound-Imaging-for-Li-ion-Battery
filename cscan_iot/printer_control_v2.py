"""
Precision 3D Printer Control Module - For C-Scan
"""
import serial
import time
import threading
import queue
import re

class PrecisionEnder:
    def __init__(self, port, baud=115200, reset_time=2.0, reset_origin=True):
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(reset_time)
        self.ser.reset_input_buffer()
        self.lock = threading.Lock()
        self.response_queue = queue.Queue()
        self._stop_reader = False
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        
        self.scan_speed_mm_s = 10.0
        self.scan_feedrate = int(self.scan_speed_mm_s * 60)
        self.positioning_feedrate = 1800
        self.line_spacing_mm = 0.1
        self.current_x = 0.0
        self.current_y = 0.0
        
        self._initialize_printer(reset_origin)
    
    def _reader_loop(self):
        buffer = ""
        while not self._stop_reader:
            try:
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting).decode('ascii', errors='ignore')
                    if data:
                        buffer += data
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip(): self.response_queue.put(line.strip())
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.1)
    
    def _initialize_printer(self, reset_origin):
        print(f"[PRINTER] Initializing (Reset Origin: {reset_origin})...")
        
        commands = [
            "G90",          # Absolute positioning
            "M83",          # Relative extrusion
            "M211 S0",      # Disable soft endstops
            "G21",          # Units: mm
            "M92 X80 Y80 Z400",
            "M203 X500 Y500 Z10 E50",
            "M204 P500 R500 T500",
            "M205 X5.0 Y5.0 Z0.4 E5.0",
        ]
        
        # Only reset origin if this is a fresh start scan
        if reset_origin:
            commands.append("G92 X0 Y0 Z0") 
        
        commands.append("M503")

        for cmd in commands:
            self._send_command_wait(cmd, timeout=3.0)
            time.sleep(0.05)
        
        print("[PRINTER] Ready.")
    
    def _send_command_wait(self, command, timeout=5.0):
        if not command.endswith('\n'): command += '\n'
        while not self.response_queue.empty():
            try: self.response_queue.get_nowait()
            except queue.Empty: break
        
        try:
            with self.lock:
                self.ser.write(command.encode('ascii'))
        except Exception as e:
            raise RuntimeError(f"Send failed: {e}")
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.response_queue.get(timeout=0.1)
                if 'ok' in resp.lower(): return resp
                elif 'error' in resp.lower() or '!!' in resp: print(f"Printer Error: {resp}")
            except queue.Empty: continue
        return "timeout"
    
    def send_command(self, command):
        if not command.endswith('\n'): command += '\n'
        with self.lock: self.ser.write(command.encode('ascii'))
    
    def wait_for_completion(self, timeout=30.0):
        self._send_command_wait("M400", timeout)
    
    def move_to_position(self, x, y, fast=True):
        f = self.positioning_feedrate if fast else self.scan_feedrate
        self.send_command(f"G1 X{x:.3f} Y{y:.3f} F{f}")
        self.current_x = x; self.current_y = y

    def move_z(self, z_mm, speed=600):
        """Moves Z axis to absolute position"""
        self.send_command(f"G1 Z{z_mm:.3f} F{speed}")
    
    def close(self):
        self.disable_motors()
        self._stop_reader = True
        if hasattr(self, 'ser') and self.ser.is_open: self.ser.close()

    def disable_motors(self):
        try: self._send_command_wait("M84", timeout=2.0)
        except: pass

def setup_precision_printer(port, baud, roi_w, roi_h, safety_margin=2.0, reset_origin=True):
    pr = PrecisionEnder(port, baud, reset_origin=reset_origin)
    x_left = safety_margin
    x_right = roi_w - safety_margin
    y_start = safety_margin
    y_end = roi_h - safety_margin
    
    if reset_origin:
        pr.move_to_position(x_left, y_start, fast=True)
        pr.wait_for_completion()
    
    return pr, x_left, x_right, y_start, y_end