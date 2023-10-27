from ncon import ncon
import operators as opr
import numpy as np


class Hamiltonian:
    def __init__(self):
        opr_obj = opr.Operator()
        self.pauli_dict = opr_obj.pauli_dictionary()
        self.spin_dict = opr_obj.spin_dictionary()

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

    def encode_hamil(
            self,
            atb: list,
            bta: list = [],
            matrix_type: str = 'pauli',
    ):
        output = {'AB': None, 'BA': None}
        blocks = self.gate_generator(atb, bta, matrix_type)

        for i in ['AB', 'BA']:
            output[i] = blocks[i][0]

        return output

