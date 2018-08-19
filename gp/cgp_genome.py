import numpy as np


class CGPGenome():
    _n_regions = None
    _length_per_region = None
    _dna = None

    def __init__(self, n_inputs, n_outputs, n_columns, n_rows, primitives):
        self._n_inputs = n_inputs
        self._n_hidden = n_columns * n_rows
        self._n_outputs = n_outputs

        self._n_columns = n_columns
        self._n_rows = n_rows

        self._primitives = primitives
        self._length_per_region = 1 + primitives.max_arity  # function gene + input genes

        # constants used as identifiers for input and output nodes
        self._id_input_node = -1
        self._id_output_node = -2
        self._non_coding_allele = None

    def randomize(self, levels_back):

        self._dna = []

        # add input nodes
        for i in range(self._n_inputs):

            # fill region with identifier for input node and zeros,
            # since input nodes do not have any inputs
            region = []
            region.append(self._id_input_node)
            region += [0] * self._primitives.max_arity

            self._dna += region

        # add hidden nodes
        for i in range(self._n_hidden):

            if i % self._n_rows == 0:  # only compute permissable inputs once per column
                column_idx = self._column_idx(i)
                permissable_inputs = self._permissable_inputs(column_idx, levels_back)

            # construct dna region consisting of function allele and
            # input alleles
            region = []
            region.append(self._primitives.sample())
            region += list(np.random.choice(permissable_inputs, self._primitives.max_arity))

            self._dna += region

        permissable_inputs = self._permissable_inputs(column_idx, levels_back, output=True)
        # add output nodes
        for i in range(self._n_outputs):

            # fill region with identifier for output node and single
            # gene determining input
            region = []
            region.append(self._id_input_node)
            region.append(np.random.choice(permissable_inputs))
            region += [0] * (self._primitives.max_arity - 1)

    # TODO: replace following two function with one that only takes
    # index of node
    def _permissable_inputs(self, column_idx, levels_back):
        permissable_inputs = []

        # all nodes can be connected to input
        permissable_inputs += [j for j in range(0, self._n_inputs)]

        # add all nodes reachable according to levels back
        lower = self._n_inputs + self._n_rows * max(0, (column_idx - levels_back))
        upper = self._n_inputs + self._n_rows * column_idx
        permissable_inputs += [j for j in range(lower, upper)]
        return permissable_inputs

    def _permissable_inputs_for_output(self):
        return self._permissable_inputs(self._n_columns, self._n_columns)

    def _random_input_for_outputs(self):
        return list(np.random.choice(range(-self._n_inputs, self._n_rows * self._n_columns), self._n_outputs))

    def __iter__(self):
        if self._dna is None:
            raise RuntimeError('dna not initialized - call CGPGenome.randomize first')
        for i in range(self._n_regions):
            yield self._dna[i * self._length_per_region:(i + 1) * self._length_per_region]

    def __getitem__(self, key):
        if self._dna is None:
            raise RuntimeError('dna not initialized - call CGPGenome.randomize first')
        return self._dna[key]

    def __len__(self):
        return self._n_regions * self._length_per_region

    @property
    def _n_regions(self):
        return self._n_inputs + self._n_hidden + self._n_outputs

    @property
    def dna(self):
        return self._dna

    @dna.setter
    def dna(self, value):

        self._check_dna_consistency(value)

        self._dna = value

    def _check_dna_consistency(self, dna):

        if len(dna) != len(self):
            raise ValueError('dna length mismatch')

        for region in self.input_regions(dna):

            if region[0] != self._id_input_node:
                raise ValueError('function genes for input nodes need to be identical to input node identifiers')

            if region[1:] != ([self._non_coding_allele] * self._primitives.max_arity):
                raise ValueError('input genes for input nodes need to be identical to non-coding allele')

        for i, region in enumerate(self.hidden_regions(dna)):

            if region[0] not in self._primitives.alleles:
                raise ValueError('function gene for hidden node has invalid value')

            column = self._column_idx(i)

            # TODO: check for levels back, currently assumes
            # levels back == n_columns
            permissable_inputs = set(self._permissable_inputs(column, self._n_columns))
            if not set(region[1:]).issubset(permissable_inputs):
                raise ValueError('input genes for hidden nodes have invalid value')

        for region in self.output_regions(dna):

            if region[0] != self._id_output_node:
                raise ValueError('function genes for output nodes need to be identical to output node identifiers')

            if region[1] not in self._permissable_inputs_for_output():
                raise ValueError('input gene for output nodes has invalid value')

            if region[2:] != [None] * (self._primitives.max_arity - 1):
                raise ValueError('non-coding input genes for output nodes need to be identical to zero')

    def _column_idx(self, hidden_region_idx):
        return hidden_region_idx // self._n_rows

    def input_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_inputs):
            yield dna[i * self._length_per_region:(i + 1) * self._length_per_region]

    def hidden_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_hidden):
            yield dna[(i + self._n_inputs) * self._length_per_region:(i + 1 + self._n_inputs) * self._length_per_region]

    def output_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_hidden):
            yield dna[(i + self._n_inputs + self._n_hidden) * self._length_per_region:(i + 1 + self._n_inputs + self._n_hidden) * self._length_per_region]

    def _is_output_gene(self, idx):
        return idx >= (self._n_regions * self._length_per_region)

    def _is_function_gene(self, idx):
        return (idx % self._length_per_region) == 0

    def mutate(self, n_mutations, levels_back):

        for i in np.random.randint(0, len(self), n_mutations):

            # TODO: parameters to control mutation rates of specific
            # genes?
            if self._is_output_gene(i):
                self._dna[i] = self._random_input_for_outputs()[0]
            elif self._is_function_gene(i):
                self._dna[i] = self._primitives.sample()
            else:
                current_column = i // self._length_per_region // self._n_rows
                permissable_inputs = self._permissable_inputs(current_column, levels_back)
                self._dna[i] = np.random.choice(permissable_inputs)
