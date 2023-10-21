from ncon import ncon
import operators as opr
import numpy as np


class Hamiltonian:
    def __init__(self):
        opr_obj = opr.Operator()
        self.pauli_dict = opr_obj.pauli_dictionary()
        self.spin_dict = opr_obj.spin_dictionary()

    def block_generator(
            self,
            map_pattern: list,
            matrix_type: str = 'pauli'
    ) -> dict:
        if matrix_type == 'pauli':
            dictionary = self.pauli_dict
        else:
            dictionary = self.spin_dict

        row_length = (len(list(map_pattern[0])))
        first_block_length = int(row_length / 2)
        index_map = []
        for i in range(1, row_length + 1):
            index_map.append([-i, -(i + row_length)])

        block_list_direct = []
        block_list_reverse = []
        for row in map_pattern:
            # map direct blocks
            explode = list(row)
            converted_direct = []
            for element in explode:
                converted_direct.append(dictionary[element])
            block_list_direct.append(ncon(converted_direct, index_map))

            # map reverse blocks
            change_side = row[first_block_length:] + row[:first_block_length]
            explode = list(change_side)
            converted_reverse = []
            for element in explode:
                converted_reverse.append(dictionary[element])
            block_list_reverse.append(ncon(converted_reverse, index_map))

        return {
            'd': block_list_direct,
            'r': block_list_reverse
        }

    def gate_generator(
            self,
            # map_pattern: dict,
            atb: list,
            bta: list = [],
            matrix_type: str = 'pauli',
            reshape_output: bool = True
    ) -> dict:
        if matrix_type == 'pauli':
            dictionary = self.pauli_dict
        else:
            dictionary = self.spin_dict

        if not bta:
            bta = atb
        map_pattern = {'AB': atb, 'BA': bta}

        row_length = len(list(map_pattern['AB'][0]))
        index_map = []
        for i in range(1, row_length + 1):
            index_map.append([-i, -(i + row_length)])

        output_blocks = {
            'AB': [],
            'BA': [],
        }
        for block in ['AB', 'BA']:
            for row in map_pattern[block]:
                explode = list(row)
                converted_direct = []
                for element in explode:
                    converted_direct.append(dictionary[element])

                if not reshape_output:
                    output_blocks[block].append(ncon(converted_direct, index_map))
                else:
                    shape_val = 2 ** int(len(list(atb[0])) / 2)
                    output_blocks[block].append(
                        ncon(converted_direct, index_map).reshape(shape_val, shape_val, shape_val, shape_val)
                    )

        return output_blocks

    def transverse_field_ising(
            self,
            j: float,
            h: float
    ) -> dict:

        atb = ['zz', 'xi', 'ix']
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            output[i] = 0.5 * -h * (blocks[i][1] + blocks[i][2]) - (j * blocks[i][0])

        return output

    def toric_code_ladder(
            self,
            av: float,
            bp: float
    ) -> dict:
        atb = ['ziizzi', 'xxiixx', 'izziiz']
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            output[i] = -av * (blocks[i][0] + blocks[i][2]) - (bp * blocks[i][1])

        return output

    def toric_code_ladder_active_x(
            self,
            av: float,
            bp: float,
            hx: float
    ) -> dict:
        atb = ['ziizzi', 'xxiixx', 'izziiz', 'xiiiii', 'ixiiii', 'iixiii']
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            output[i] = -av * (blocks[i][0] + blocks[i][2]) - (bp * blocks[i][1]) - (hx * (
                    blocks[i][3] + blocks[i][4] + blocks[i][5]))

        return output

    def toric_code_ladder_active_z(
            self,
            av: float,
            bp: float,
            hz: float
    ) -> dict:
        atb = ['ziizzi', 'xxiixx', 'izziiz', 'ziiiii', 'iziiii', 'iiziii']
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            output[i] = -av * (blocks[i][0] + blocks[i][2]) - (bp * blocks[i][1]) - (hz * (
                    blocks[i][3] + blocks[i][4] + blocks[i][5]))

        return output

    def toric_code_ladder_active_xz(
            self,
            av: float,
            bp: float,
            hx: float,
            hz: float
    ) -> dict:
        atb = [
            'ziizzi', 'xxiixx', 'izziiz',
            'xiiiii', 'ixiiii', 'iixiii',
            'ziiiii', 'iziiii', 'iiziii',
        ]
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            output[i] = -av * (blocks[i][0] + blocks[i][2]) - (bp * blocks[i][1]) - (
                    hx * (blocks[i][3] + blocks[i][4] + blocks[i][5])) - (
                                hz * (blocks[i][6] + blocks[i][7] + blocks[i][8]))

        return output

    def star_a_c(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['ziizzi'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def star_d_f(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['izziiz'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def plaquette_b_e(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['xxiixx'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_1(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['xiiiii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_2(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['ixiiii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_3(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iixiii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_4(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiixii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_5(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiiixi'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_x_6(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiiiix'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_1(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['ziiiii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_2(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iziiii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_3(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiziii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_4(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiizii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_5(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiiizi'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def magnetization_z_6(self) -> dict:
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iiiiiz'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def question(self, hx, hy, hz) -> dict:
        atb = [
            'zzii', 'iyyi', 'xixi',
            'xiii', 'ixii', 'iixi',
            'yiii', 'iyii', 'iiyi',
            'ziii', 'izii', 'iizi',
        ]
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb)

        for i in ['AB', 'BA']:
            # output[i] = blocks[i][0] - blocks[i][1] - blocks[i][2]
            output[i] = blocks[i][0] + blocks[i][1] + blocks[i][2] - (
                    hx * (blocks[i][3] + blocks[i][4] + blocks[i][5])) - (
                                hy * (blocks[i][6] + blocks[i][7] + blocks[i][8])) - (
                                    hz * (blocks[i][9] + blocks[i][10] + blocks[i][11]))

        return output

    def czz(self):
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['zzii'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def cyy(self):
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['iyyi'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

    def cxx(self):
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(['xixi'])

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

