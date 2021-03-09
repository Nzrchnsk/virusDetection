import struct

def file_to_bytes(file_path: str) -> [bytes]:
    file = open(file_path, "rb")
    byte = file.read(1)
    byte_arr = []
    while byte:
        byte = file.read(1)
        # TODO: convert to float?
        if byte:
            unpack_byte = struct.unpack('B', byte)
        byte_arr.append(unpack_byte[0])
        # byte_arr.append(byte)
    file.close()
    return byte_arr
