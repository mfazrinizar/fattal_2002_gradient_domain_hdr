import numpy as np

class HDRLoaderResult:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.cols = None

class HDRLoader:
    MINELEN = 8
    MAXELEN = 0x7fff
    
    @staticmethod
    def load(filename):
        result = HDRLoaderResult()
        
        try:
            with open(filename, 'rb') as file:
                # Check header
                header = file.read(10)
                if header != b'#?RADIANCE':
                    return False, result
                
                file.seek(1, 1)  # SEEK_CUR
                
                # Read commands
                cmd = []
                c = b'\x00'
                oldc = b'\x00'
                while True:
                    oldc = c
                    c = file.read(1)
                    if c == b'\n' and oldc == b'\n':
                        break
                    cmd.append(c)
                
                # Read resolution
                reso = []
                while True:
                    c = file.read(1)
                    reso.append(c)
                    if c == b'\n':
                        break
                
                reso_str = b''.join(reso).decode('ascii')
                parts = reso_str.split()
                h = int(parts[1])
                w = int(parts[3])
                
                result.width = w
                result.height = h
                
                cols = np.zeros((h * w * 3,), dtype=np.float32)
                
                # Convert image
                for y in range(h - 1, -1, -1):
                    scanline = HDRLoader.decrunch(w, file)
                    if scanline is None:
                        break
                    HDRLoader.work_on_rgbe(scanline, w, cols, (h - 1 - y) * w * 3)
                
                result.cols = cols
                return True, result
                
        except Exception as e:
            print(f"Error loading HDR: {e}")
            return False, result
    
    @staticmethod
    def convert_component(expo, val):
        v = val / 256.0
        d = 2.0 ** expo
        return v * d
    
    @staticmethod
    def work_on_rgbe(scan, length, cols, offset):
        for i in range(length):
            # Convert to signed integer properly
            expo = int(scan[i][3]) - 128
            cols[offset + i * 3 + 0] = HDRLoader.convert_component(expo, int(scan[i][0]))
            cols[offset + i * 3 + 1] = HDRLoader.convert_component(expo, int(scan[i][1]))
            cols[offset + i * 3 + 2] = HDRLoader.convert_component(expo, int(scan[i][2]))
    
    @staticmethod
    def decrunch(length, file):
        if length < HDRLoader.MINELEN or length > HDRLoader.MAXELEN:
            return HDRLoader.old_decrunch(length, file)
        
        i = ord(file.read(1))
        if i != 2:
            file.seek(-1, 1)
            return HDRLoader.old_decrunch(length, file)
        
        scanline = np.zeros((length, 4), dtype=np.uint8)
        scanline[0][1] = ord(file.read(1))
        scanline[0][2] = ord(file.read(1))
        i = ord(file.read(1))
        
        if scanline[0][1] != 2 or scanline[0][2] & 128:
            scanline[0][0] = 2
            scanline[0][3] = i
            return HDRLoader.old_decrunch(length - 1, file, scanline, 1)
        
        for i in range(4):
            j = 0
            while j < length:
                code = ord(file.read(1))
                if code > 128:
                    code &= 127
                    val = ord(file.read(1))
                    for _ in range(code):
                        scanline[j][i] = val
                        j += 1
                else:
                    for _ in range(code):
                        scanline[j][i] = ord(file.read(1))
                        j += 1
        
        return scanline
    
    @staticmethod
    def old_decrunch(length, file, scanline=None, start_idx=0):
        if scanline is None:
            scanline = np.zeros((length, 4), dtype=np.uint8)
        
        i = start_idx
        rshift = 0
        
        while i < length:
            try:
                r = ord(file.read(1))
                g = ord(file.read(1))
                b = ord(file.read(1))
                e = ord(file.read(1))
            except:
                return scanline
            
            if r == 1 and g == 1 and b == 1:
                for _ in range(e << rshift):
                    if i > 0:
                        scanline[i] = scanline[i - 1]
                    i += 1
                    if i >= length:
                        break
                rshift += 8
            else:
                scanline[i] = [r, g, b, e]
                i += 1
                rshift = 0
        
        return scanline