import operators as opr
from ncon import ncon


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

    def transverse_field_ising(
            self,
            j: float,
            h: float
    ) -> dict:
        map_pattern = [
            'zz',
            'xi',
            'ix',
        ]

        blocks = self.block_generator(map_pattern)
        output = {
            'd': None,
            'r': None
        }
        for i in ['d', 'r']:
            first_term = -j * blocks[i][0]
            second_term = -h * 0.5 * (blocks[i][1] + blocks[i][2])
            output[i] = (first_term + second_term).reshape(2, 2, 2, 2)

        return output

    def toric_code_ribbon(
            self,
            av: float,
            bp: float
    ) -> dict:
        map_pattern = [
            'ziizzi',
            'xxiixx',
            'izziiz',
            'zzizii',
            'ixxxxi',
            'iizizz'
        ]

        blocks = self.block_generator(map_pattern)
        output = {
            'd': None,
            'r': None
        }
        for i in ['d', 'r']:
            first_term = -av * (blocks[i][0] + blocks[i][2] + blocks[i][3] + blocks[i][5])
            second_term = -bp * (blocks[i][1] + blocks[i][4])
            output[i] = (first_term + second_term).reshape(8, 8, 8, 8)

        return output

    def toric_code_ribbon_II(
            self,
            av: float,
            bp: float
    ) -> dict:
        map_pattern = [
            'zizz',
            'zzzi',
            'xxix',
            'xiii',
            'ixxx',
            'iixi',
        ]

        blocks = self.block_generator(map_pattern)
        output = {
            'd': None,
            'r': None
        }
        for i in ['d', 'r']:
            first_term = -av * (blocks[i][2] + blocks[i][3] + blocks[i][4] + blocks[i][5])
            second_term = -bp * (blocks[i][0] + blocks[i][1])
            output[i] = (first_term + second_term).reshape(4, 4, 4, 4)

        return output
