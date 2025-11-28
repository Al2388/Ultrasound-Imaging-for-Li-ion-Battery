"""
精准3D打印机控制模块 - 用于C扫描
确保精确的蛇形扫描：恒定1cm/s速度，0.1mm行距
"""
import serial
import time
import threading
import queue
import re
import math

class PrecisionEnder:
    def __init__(self, port, baud=115200, reset_time=2.0):
        """初始化打印机连接"""
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(reset_time)
        self.ser.reset_input_buffer()
        self.lock = threading.Lock()
        self.response_queue = queue.Queue()
        self._stop_reader = False
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        
        # 扫描参数
        self.scan_speed_mm_s = 10.0  # 1cm/s = 10mm/s
        self.scan_feedrate = int(self.scan_speed_mm_s * 60)  # 600 mm/min
        self.positioning_feedrate = 1800  # 30mm/s for positioning moves
        self.line_spacing_mm = 0.1  # 0.1mm行距
        
        # 当前位置跟踪
        self.current_x = 0.0
        self.current_y = 0.0
        
        # 初始化打印机设置
        self._initialize_printer()
    
    def _reader_loop(self):
        """后台线程读取打印机响应"""
        buffer = ""
        while not self._stop_reader:
            try:
                data = self.ser.read(1).decode('ascii', errors='ignore')
                if data:
                    buffer += data
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        buffer = lines[-1]  # 保留未完成的行
                        for line in lines[:-1]:
                            line = line.strip()
                            if line:
                                self.response_queue.put(line)
                else:
                    time.sleep(0.001)  # 1ms pause to prevent CPU spinning
            except Exception as e:
                if not self._stop_reader:
                    print(f"[PRINTER] Reader error: {e}")
                time.sleep(0.1)
    
    def _initialize_printer(self):
        """初始化打印机设置"""
        print("[PRINTER] Initializing printer settings...")
        
        # 基本设置
        commands = [
            "G90",  # 绝对坐标模式
            "M83",  # 相对挤出模式（虽然不使用挤出，但确保安全）
            "M211 S0",  # 禁用软件限位（小心使用）
            "G21",  # 单位：毫米
            
            # 步数设置 (Ender 3 默认值)
            "M92 X80 Y80 Z400 E93",
            
            # 精确的运动参数设置
            "M203 X500 Y500 Z10 E50",  # 最大速度 (mm/min)
            "M204 P300 R300 T300",     # 加速度设置 300 mm/s²
            "M205 X8.0 Y8.0 Z0.4 E5.0",  # 抖动控制
            
            # 设置当前位置为原点
            "G92 X0 Y0 Z0",
            
            # 确认设置
            "M503",  # 显示当前设置
        ]
        
        for cmd in commands:
            self._send_command_wait(cmd, timeout=3.0)
            time.sleep(0.1)  # 命令间延迟
        
        print("[PRINTER] Printer initialized successfully")
    
    def _send_command_wait(self, command, timeout=5.0):
        """发送命令并等待OK响应"""
        if not command.endswith('\n'):
            command += '\n'
        
        # 清空响应队列
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break
        
        # 发送命令
        try:
            with self.lock:
                self.ser.write(command.encode('ascii'))
                self.ser.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to send command '{command.strip()}': {e}")
        
        # 等待响应
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if 'ok' in response.lower() or 'done' in response.lower():
                    return response
                elif 'error' in response.lower():
                    raise RuntimeError(f"Printer error: {response}")
            except queue.Empty:
                continue
        
        raise TimeoutError(f"Command '{command.strip()}' timed out after {timeout}s")
    
    def send_command(self, command):
        """发送简单命令（不等待响应，用于流式控制）"""
        if not command.endswith('\n'):
            command += '\n'
        try:
            with self.lock:
                self.ser.write(command.encode('ascii'))
                self.ser.flush()
        except Exception as e:
            print(f"[PRINTER] Command send error: {e}")
            raise
    
    def wait_for_completion(self, timeout=30.0):
        """等待所有运动完成"""
        try:
            self._send_command_wait("M400", timeout)  # 等待运动完成
        except TimeoutError:
            print("[PRINTER] Warning: Motion completion timeout")
    
    def move_to_position(self, x, y, fast=True):
        """精确移动到指定位置"""
        feedrate = self.positioning_feedrate if fast else self.scan_feedrate
        command = f"G1 X{x:.3f} Y{y:.3f} F{feedrate}"
        self.send_command(command)
        self.current_x = x
        self.current_y = y
        print(f"[PRINTER] Moving to X{x:.3f} Y{y:.3f} at {feedrate}mm/min")
    
    def scan_line(self, x_start, x_end, y, direction='positive'):
        """执行精确的扫描线"""
        # 首先移动到起始位置（快速定位）
        self.move_to_position(x_start, y, fast=True)
        self.wait_for_completion()
        
        # 执行扫描移动（恒定速度）
        print(f"[PRINTER] Scanning line Y{y:.3f} from X{x_start:.3f} to X{x_end:.3f} at {self.scan_speed_mm_s}mm/s")
        command = f"G1 X{x_end:.3f} F{self.scan_feedrate}"
        self.send_command(command)
        self.current_x = x_end
        
        # 计算理论移动时间
        distance = abs(x_end - x_start)
        theoretical_time = distance / self.scan_speed_mm_s
        return theoretical_time
    
    def execute_serpentine_scan(self, x_min, x_max, y_start, num_lines, line_spacing=None):
        """执行完整的蛇形扫描"""
        if line_spacing is None:
            line_spacing = self.line_spacing_mm
        
        print(f"[PRINTER] Starting serpentine scan:")
        print(f"  X range: {x_min:.3f} to {x_max:.3f} mm")
        print(f"  Y start: {y_start:.3f} mm")
        print(f"  Lines: {num_lines}")
        print(f"  Line spacing: {line_spacing:.3f} mm")
        print(f"  Scan speed: {self.scan_speed_mm_s} mm/s")
        
        scan_times = []
        
        for line_num in range(num_lines):
            y_pos = y_start + line_num * line_spacing
            
            # 蛇形扫描：奇数行从左到右，偶数行从右到左
            if line_num % 2 == 0:
                # 左到右
                scan_time = self.scan_line(x_min, x_max, y_pos)
                print(f"[SCAN] Line {line_num + 1}/{num_lines}: L→R at Y{y_pos:.3f}")
            else:
                # 右到左
                scan_time = self.scan_line(x_max, x_min, y_pos)
                print(f"[SCAN] Line {line_num + 1}/{num_lines}: R→L at Y{y_pos:.3f}")
            
            scan_times.append(scan_time)
            
            # 等待当前行完成
            self.wait_for_completion()
        
        return scan_times
    
    def home_axes(self, axes='XY'):
        """归零指定轴"""
        print(f"[PRINTER] Homing {axes} axes...")
        if 'X' in axes.upper():
            self._send_command_wait("G28 X", timeout=30.0)
        if 'Y' in axes.upper():
            self._send_command_wait("G28 Y", timeout=30.0)
        if 'Z' in axes.upper():
            self._send_command_wait("G28 Z", timeout=30.0)
        
        # 更新位置跟踪
        if 'X' in axes.upper():
            self.current_x = 0.0
        if 'Y' in axes.upper():
            self.current_y = 0.0
    
    def get_position(self):
        """获取当前位置"""
        try:
            response = self._send_command_wait("M114", timeout=3.0)
            # 解析位置信息 "X:12.34 Y:56.78 Z:9.10"
            x_match = re.search(r'X:?(-?\d+\.?\d*)', response)
            y_match = re.search(r'Y:?(-?\d+\.?\d*)', response)
            
            if x_match:
                self.current_x = float(x_match.group(1))
            if y_match:
                self.current_y = float(y_match.group(1))
            
            return self.current_x, self.current_y
        except Exception as e:
            print(f"[PRINTER] Position query failed: {e}")
            return self.current_x, self.current_y
    
    def emergency_stop(self):
        """紧急停止"""
        print("[PRINTER] EMERGENCY STOP!")
        try:
            self.send_command("M112")  # 紧急停止
        except:
            pass
    
    def disable_motors(self):
        """禁用电机"""
        try:
            self._send_command_wait("M84", timeout=3.0)
            print("[PRINTER] Motors disabled")
        except Exception as e:
            print(f"[PRINTER] Failed to disable motors: {e}")
    
    def close(self):
        """关闭打印机连接"""
        print("[PRINTER] Shutting down printer connection...")
        self.disable_motors()
        self._stop_reader = True
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        print("[PRINTER] Printer connection closed")


def setup_precision_printer(port, baud, roi_w_mm, roi_h_mm, safety_margin=2.0):
    """设置精密打印机控制"""
    pr = PrecisionEnder(port, baud)
    
    # 定义扫描区域边界
    x_left = safety_margin
    x_right = roi_w_mm - safety_margin
    y_start = safety_margin
    y_end = roi_h_mm - safety_margin
    
    # 验证扫描区域
    scan_width = x_right - x_left
    scan_height = y_end - y_start
    
    if scan_width <= 0 or scan_height <= 0:
        raise ValueError(f"Invalid scan area: width={scan_width:.3f}mm, height={scan_height:.3f}mm")
    
    print(f"[PRINTER] Scan area: {scan_width:.3f} x {scan_height:.3f} mm")
    print(f"  X: {x_left:.3f} to {x_right:.3f} mm")
    print(f"  Y: {y_start:.3f} to {y_end:.3f} mm")
    
    # 移动到起始位置
    pr.move_to_position(x_left, y_start, fast=True)
    pr.wait_for_completion()
    
    return pr, x_left, x_right, y_start, y_end