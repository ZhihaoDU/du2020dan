import numpy as np
import struct


class KaldiFeatHolder(object):
    def __init__(self, key_len, max_frame_number, feat_dim):
        self.key_len = key_len
        self.magic_len = 6
        self.token_len = (1+4)*2
        self.header_len = self.key_len + self.magic_len + self.token_len
        self.max_frame_number = max_frame_number
        self.fead_dim = feat_dim
        self.file_content = bytearray(self.header_len + 4 * max_frame_number * feat_dim)
        self.key = None
        self.offset = 0
        self.len = 0
        self.value_len = 0

    def set_key(self, key):
        if len(key) != self.key_len:
            print("Fatal Error: The length of key %s is illegal, except %d, but %d." % (key, self.key_len, len(key)))
            exit(-101)
        self.key = key
        self.offset = 0
        self.file_content[self.offset: self.offset + self.key_len] = bytearray(self.key, encoding='ascii')
        self.offset += self.key_len
        self.file_content[self.offset: self.offset + self.magic_len] = b' \x00BFM '
        self.offset += self.magic_len
        self.len = self.key_len + self.magic_len

    def set_value(self, value):
        value = np.array(value, dtype=np.float32)
        if len(value.shape) != 2:
            print("Fatal Error: The feature matrix must be 2 dim.")
            exit(-201)
        rows, cols = value.shape
        if rows > self.max_frame_number or cols != self.fead_dim:
            print("Fatal Error: The shape of feature matrix mismatch with this holder, except(<=%d, %d), but (%d, %d)." %
                  (self.max_frame_number, self.fead_dim, rows, cols))
            print("Tips: The feature matrix must be in [frames, feat_dim] shape, aka. each frame each row.")
            exit(-202)
        self.offset = self.key_len + self.magic_len
        self.file_content[self.offset] = 4
        self.file_content[self.offset + 1: self.offset + 5] = struct.pack('i', rows)
        self.file_content[self.offset + 5] = 4
        self.file_content[self.offset + 6: self.offset + 10] = struct.pack('i', cols)
        self.offset += self.token_len
        self.value_len = 4 * rows * cols
        fv = value.flatten()
        self.file_content[self.offset: self.offset+self.value_len] = struct.pack('f'*rows*cols, *fv)
        self.offset += self.value_len
        self.len = self.offset

    def __len__(self):
        return self.len

    def get_real_len(self):
        return self.len

    def write_to(self, file):
        file.write(self.file_content[:self.len])


if __name__ == '__main__':
    feat = [[1.0, 2.0], [3.0, 4.0]]
    holder = KaldiFeatHolder(8, 4, 2)
    holder.set_key('00000001')
    holder.set_value(feat)
    ark = open('test/b2.bin', 'wb')
    scp = open('test/b2.scp', 'w')
    holder.write_to(ark)
    scp.write('%s %s:%d\n' % ('00000001', 'b2.scp', 9))
    scp.write('%s %s:%d' % ('00000001', 'b2.scp', holder.len+9))
    feat = [[5.0, 6.0], [7.0, 8.0]]
    holder.set_key('00000002')
    holder.set_value(feat)
    holder.write_to(ark)
    ark.close()


