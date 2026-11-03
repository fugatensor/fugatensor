"""Independent CUDA parameter probe: mixed widths, pointer padding and null bits."""

SAMPLES = (
    ('i1', True, 1), ('i8', -5, 251), ('i16', -1234, 64302),
    ('i32', -7, 4294967289), ('i64', -9, -9), ('u8', 255, 255),
    ('u16', 65535, 65535), ('u32', 2**32 - 1, 2**32 - 1),
    ('u64', 2**64 - 1, -1), ('fp16', 1.5, 0x3e00),
    ('bf16', -2.5, 0xc020), ('fp32', 1.25, 0x3fa00000),
    ('fp64', -0.0, -(2**63)),
)
WIDTHS = {'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
          'u8': 1, 'u16': 2, 'u32': 4, 'u64': 8,
          'fp16': 2, 'bf16': 2, 'fp32': 4, 'fp64': 8}


def scalar_probe():
    parameters = ['.param .u64 input_ptr']
    arguments = [{'argument': 0, 'kind': 'pointer', 'binding': 'input:0'}]
    body = ['ld.param.u64 %out, [output_ptr];']
    for index, (dtype, value, _) in enumerate(SAMPLES):
        if index == 2:
            output_index = len(arguments)
            parameters.append('.param .u64 output_ptr')
            arguments.append({'argument': output_index, 'kind': 'pointer', 'binding': 'output:0'})
        width = WIDTHS[dtype]
        parameters.append(f'.param .u{width * 8} scalar{index}')
        arguments.append({'argument': len(arguments), 'kind': 'scalar', 'type': dtype, 'value': value})
        if width == 8:
            body.append(f'ld.param.u64 %wide, [scalar{index}];')
        else:
            body.extend([f'ld.param.u{width * 8} %small, [scalar{index}];', 'cvt.u64.u32 %wide, %small;'])
        body.append(f'st.global.u64 [%out+{index * 8}], %wide;')
    parameters.append('.param .u64 null_ptr')
    arguments.append({'argument': len(arguments), 'kind': 'pointer', 'binding': 'null'})
    body.extend(['ld.param.u64 %wide, [null_ptr];', f'st.global.u64 [%out+{len(SAMPLES) * 8}], %wide;'])
    payload = ('.version 8.0\n.target sm_80\n.address_size 64\n.visible .entry scale2(' +
               ', '.join(parameters) + ') {\n.reg .b32 %small;\n.reg .b64 %wide, %out;\n' +
               '\n'.join(body) + '\nret;\n}\n')
    return payload, arguments, output_index, [value for _, _, value in SAMPLES] + [0]
