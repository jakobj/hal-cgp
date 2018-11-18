import numpy as np


class CGPGenome():
    _n_regions = None
    _length_per_region = None
    _dna = None

    def __init__(self, n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives):
        self._n_inputs = n_inputs
        self._n_hidden = n_columns * n_rows
        self._n_outputs = n_outputs

        self._n_columns = n_columns
        self._n_rows = n_rows

        if levels_back == 0:
            raise ValueError('levels_back must be strictly positive')
        if levels_back > n_columns:
            raise ValueError('levels_back can not be larger than n_columns')
        self._levels_back = levels_back

        self._primitives = primitives
        self._length_per_region = 1 + primitives.max_arity  # function gene + input genes

        # constants used as identifiers for input and output nodes
        self._id_input_node = -1
        self._id_output_node = -2
        self._non_coding_allele = None

    def __repr__(self):
        s = self.__class__.__name__ + '('
        for region_idx, input_region in self.input_regions():
            s += str(region_idx) + ': ' + str(input_region) + ' | '
        for region_idx, hidden_region in self.hidden_regions():
            s += str(region_idx) + ': ' + str(hidden_region) + ' | '
        for region_idx, output_region in self.output_regions():
            s += str(region_idx) + ': ' + str(output_region) + ' | '
        s = s[:-3]
        s += ')'
        return s

    def randomize(self):

        dna = []

        # add input regions
        for i in range(self._n_inputs):

            # fill region with identifier for input node and zeros,
            # since input nodes do not have any inputs
            region = []
            region.append(self._id_input_node)
            region += [self._non_coding_allele] * self._primitives.max_arity

            dna += region

        # add hidden nodes
        for i in range(self._n_hidden):

            if i % self._n_rows == 0:  # only compute permissable inputs once per column
                permissable_inputs = self._permissable_inputs(i + self._n_inputs)

            # construct dna region consisting of function allele and
            # input alleles
            region = []
            node_id = self._primitives.sample()
            region.append(node_id)

            # choose inputs for coding region
            region += list(np.random.choice(permissable_inputs, self._primitives[node_id]._arity))

            # mark non-coding region
            region += [self._non_coding_allele] * (self._primitives.max_arity - self._primitives[node_id]._arity)

            dna += region

        permissable_inputs = self._permissable_inputs_for_output_region()
        # add output nodes
        for i in range(self._n_outputs):

            # fill region with identifier for output node and single
            # gene determining input
            region = []
            region.append(self._id_output_node)
            region.append(np.random.choice(permissable_inputs))
            region += [self._non_coding_allele] * (self._primitives.max_arity - 1)

            dna += region

        self._validate_dna(dna)

        self._dna = dna

    def _permissable_inputs(self, region_idx):

        assert not self._is_input_region(region_idx)

        permissable_inputs = []

        # all nodes can be connected to input
        permissable_inputs += [j for j in range(0, self._n_inputs)]

        # add all nodes reachable according to levels back
        if self._is_hidden_region(region_idx):
            hidden_column_idx = self._hidden_column_idx(region_idx)
            lower = self._n_inputs + self._n_rows * max(0, (hidden_column_idx - self._levels_back))
            upper = self._n_inputs + self._n_rows * hidden_column_idx
        else:
            assert self._is_output_region(region_idx)
            lower = self._n_inputs
            upper = self._n_inputs + self._n_rows * self._n_columns

        permissable_inputs += [j for j in range(lower, upper)]

        return permissable_inputs

    def _permissable_inputs_for_output_region(self):
        return self._permissable_inputs(self._n_inputs + self._n_rows * self._n_columns)

    def __iter__(self):
        if self._dna is None:
            raise RuntimeError('dna not initialized')
        for i in range(self._n_regions):
            yield self._dna[i * self._length_per_region:(i + 1) * self._length_per_region]

    def __getitem__(self, key):
        if self._dna is None:
            raise RuntimeError('dna not initialized')
        return self._dna[key]

    def __setitem__(self, key, value):
        dna = list(self._dna)
        dna[key] = value
        self._validate_dna(dna)
        self._dna = dna

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

        self._validate_dna(value)

        self._dna = value

    def _validate_dna(self, dna):

        if len(dna) != len(self):
            raise ValueError('dna length mismatch')

        for region_idx, input_region in self.input_regions(dna):

            if input_region[0] != self._id_input_node:
                raise ValueError('function genes for input nodes need to be identical to input node identifiers')

            if input_region[1:] != ([self._non_coding_allele] * self._primitives.max_arity):
                raise ValueError('input genes for input nodes need to be identical to non-coding allele')

        for region_idx, hidden_region in self.hidden_regions(dna):

            if hidden_region[0] not in self._primitives.alleles:
                raise ValueError('function gene for hidden node has invalid value')

            coding_input_genes = hidden_region[1:self._primitives[hidden_region[0]]._arity + 1]
            non_coding_input_genes = hidden_region[self._primitives[hidden_region[0]]._arity + 1:]

            permissable_inputs = set(self._permissable_inputs(region_idx))
            if not set(coding_input_genes).issubset(permissable_inputs):
                raise ValueError('input genes for hidden nodes have invalid value')

            if non_coding_input_genes != [self._non_coding_allele] * (self._primitives.max_arity - self._primitives[hidden_region[0]]._arity):
                raise ValueError('non-coding input genes for hidden nodes need to be identical to non-coding allele')

        for region_idx, output_region in self.output_regions(dna):

            if output_region[0] != self._id_output_node:
                raise ValueError('function genes for output nodes need to be identical to output node identifiers')

            if output_region[1] not in self._permissable_inputs_for_output_region():
                raise ValueError('input gene for output nodes has invalid value')

            if output_region[2:] != [None] * (self._primitives.max_arity - 1):
                raise ValueError('non-coding input genes for output nodes need to be identical to non-coding allele')

    def _hidden_column_idx(self, region_idx):
        assert self._n_inputs <= region_idx
        assert region_idx < self._n_inputs + self._n_rows * self._n_columns
        hidden_column_idx = (region_idx - self._n_inputs) // self._n_rows
        assert 0 <= hidden_column_idx
        assert hidden_column_idx < self._n_columns
        return hidden_column_idx

    def input_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_inputs):
            region_idx = i
            region = dna[region_idx * self._length_per_region:(region_idx + 1) * self._length_per_region]
            yield region_idx, region

    def hidden_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_hidden):
            region_idx = i + self._n_inputs
            region = dna[region_idx * self._length_per_region:(region_idx + 1) * self._length_per_region]
            yield region_idx, region

    def output_regions(self, dna=None):
        if dna is None:
            dna = self.dna
        for i in range(self._n_outputs):
            region_idx = i + self._n_inputs + self._n_hidden
            region = dna[region_idx * self._length_per_region:(region_idx + 1) * self._length_per_region]
            yield region_idx, region

    def _is_input_region(self, region_idx):
        return region_idx < self._n_inputs

    def _is_hidden_region(self, region_idx):
        return (self._n_inputs <= region_idx) & (region_idx < self._n_inputs + self._n_hidden)

    def _is_output_region(self, region_idx):
        return self._n_inputs + self._n_hidden <= region_idx

    def _is_function_gene(self, idx):
        return (idx % self._length_per_region) == 0

    def _is_input_gene(self, idx):
        return not self._is_function_gene(idx)

    def mutate(self, n_mutations):

        assert isinstance(n_mutations, int) and 0 < n_mutations

        successful_mutations = 0
        while successful_mutations < n_mutations:

            gene_idx = np.random.randint(0, len(self))

            region_idx = gene_idx // self._length_per_region

            # TODO: parameters to control mutation rates of specific
            # genes?
            if self._is_input_region(region_idx):
                continue  # nothing to do here

            elif self._is_output_region(region_idx):
                success = self._mutate_output_region(gene_idx, region_idx)
                if success:
                    successful_mutations += 1

            else:
                success = self._mutate_hidden_region(gene_idx, region_idx)
                if success:
                    successful_mutations += 1

        self._validate_dna(self._dna)

    def _mutate_output_region(self, gene_idx, region_idx):
        assert(self._is_output_region(region_idx))

        # only mutate coding output gene
        if self._is_input_gene(gene_idx) and self._dna[gene_idx] is not self._non_coding_allele:
            permissable_inputs = self._permissable_inputs_for_output_region()
            self._dna[gene_idx] = np.random.choice(permissable_inputs)
            return True

        return False

    def _mutate_hidden_region(self, gene_idx, region_idx):
        assert(self._is_hidden_region(region_idx))

        if self._is_function_gene(gene_idx):
            self._dna[gene_idx] = self._primitives.sample()

            # since we have changed the function gene, we need
            # to update the input genes to match the arity of
            # the new function

            # first: set coding genes to valid input allele
            permissable_inputs = self._permissable_inputs(region_idx)

            for j in range(1, 1 + self._primitives[self._dna[gene_idx]]._arity):
                self._dna[gene_idx + j] = np.random.choice(permissable_inputs)

            # second: set non-coding genes to non-coding allele
            for j in range(1 + self._primitives[self._dna[gene_idx]]._arity, 1 + self._primitives.max_arity):
                self._dna[gene_idx + j] = self._non_coding_allele

            return True

        else:
            if self._dna[gene_idx] is not self._non_coding_allele:
                permissable_inputs = self._permissable_inputs(region_idx)
                self._dna[gene_idx] = np.random.choice(permissable_inputs)
                return True

        return False

    @property
    def primitives(self):
        return self._primitives

    def clone(self):
        new = CGPGenome(self._n_inputs, self._n_outputs, self._n_columns, self._n_rows, self._levels_back, self._primitives)
        new.dna = self._dna.copy()
        return new
