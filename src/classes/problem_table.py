import pandas as pd
import numpy as np
from copy import deepcopy
from ..lib.enums import ProblemTableLabel

class ProblemTable:
    def __init__(self, data_input: dict | list = None, add_default_labels: bool = True):
        if add_default_labels:
            self.columns = list([index.value for index in ProblemTableLabel])
        else:
            self.columns = list([key for key in data_input.keys()])
            
            for key in self.columns:
                if np.isnan(data_input[key]).all():
                    data_input.pop(key)
        self.col_index = {col: idx for idx, col in enumerate(self.columns)}

        if isinstance(data_input, dict):
            # Align data from dict into array using columns order
            self.data = np.array([
                data_input.get(col, [np.nan] * len(next(iter(data_input.values()))))
                for col in self.columns
            ]).T
        elif isinstance(data_input, list):
            data_input = self._pad_data_input(data_input, len(self.columns))
            self.data = np.array(data_input).T
        else:
            self.data = None

    class ColumnViewByIndex:
        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, idx):
            return self.parent.data[:, idx]

        def __setitem__(self, idx, values):
            self.parent.data[:, idx] = values
    
    @property
    def icol(self):
        return self.ColumnViewByIndex(self)   
    
    class ColumnViewByName:
        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_name):
            idx = self.parent.col_index[col_name]
            return self.parent.data[:, idx]

        def __setitem__(self, col_name, values):
            idx = self.parent.col_index[col_name]
            if self.parent.data is not None:
                self.parent.data[:, idx] = values
            else:
                data_input = {col_name: values}
                self.data = np.array([
                    data_input.get(col, [np.nan] * len(next(iter(data_input.values()))))
                    for col in self.parent.columns
                ]).T

    @property
    def col(self):
        return self.ColumnViewByName(self)  
    
    class ColumnsViewByName:
        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, col_names):
            idxs = []
            for col_name in col_names:
                idxs.append(self.parent.col_index[col_name])
            return self.parent.data[:, idxs]

        def __setitem__(self, col_name, values):
            idx = self.parent.col_index[col_name]
            if self.parent.data is not None:
                self.parent.data[:, idx] = values
            else:
                data_input = {col_name: values}
                self.data = np.array([
                    data_input.get(col, [np.nan] * len(next(iter(data_input.values()))))
                    for col in self.parent.columns
                ]).T

    @property
    def cols(self):
        return self.ColumnsViewByName(self)   
    
    class LocationByRowByColName:
        def __init__(self, parent: "ProblemTable"):
            self.parent = parent
        
        def __getitem__(self, key):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            return self.parent.data[row_idx, col_idx]
        
        def __setitem__(self, key, value):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            self.parent.data[row_idx, col_idx] = value    

    @property
    def loc(self):
        return self.LocationByRowByColName(self)    
    
    class LocationByRowByCol:
        def __init__(self, parent: "ProblemTable"):
            self.parent = parent

        def __getitem__(self, key):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            return self.parent.data[row_idx, col_idx]
        
        def __setitem__(self, key, value):
            row_idx, col_key = key
            col_idx = self.parent.col_index[col_key]
            self.parent.data[row_idx, col_idx] = value  
  
    @property
    def iloc(self):
        return self.LocationByRowByCol(self)     

    def __len__(self):
        if isinstance(self.data, np.ndarray): 
            return self.data.shape[0]
        else:
            return 0

    def __getitem__(self, keys):
        data_input = {}
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            data_input[key] = self.col[key]
        return ProblemTable(
            data_input, add_default_labels=False
        )

    def __eq__(self, other):
        if not isinstance(other, ProblemTable):
            return False
        if self.columns != other.columns:
            return False
        if self.data.shape != other.data.shape:
            return False

        # NaN-safe elementwise comparison
        a = self.data
        b = other.data
        nan_mask = np.isnan(a) & np.isnan(b)
        close_mask = np.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)
        return np.all(nan_mask | close_mask)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the buffer into a pandas DataFrame."""
        return pd.DataFrame(self.data.copy, columns=self.columns)

    @property
    def copy(self):
        return deepcopy(self)
    
    def _pad_data_input(self, data_input, n_cols):
        current_cols = len(data_input)
        if current_cols < n_cols:
            n_rows = len(data_input[0])  # assume all rows are same length
            padding = [[np.nan] * n_rows for _ in range(n_cols - current_cols)]
            data_input += padding
        return data_input
    
    def to_list(self, col: str = None):
        if isinstance(col, str):
            ls = self.col[col].T.tolist()
        elif col == None:
            ls = self.data.T.tolist()
        return ls[0] if len(ls) == 1 else ls

    def delta_col(self, key, shift: int =1) -> np.ndarray:
        idx = self.col_index[key]
        col_values = self.data[:, idx]
        delta = np.roll(col_values, shift) - col_values
        delta[0] = 0.0
        return delta
    
    def shift(self, key, shift: int =1, filler_value: float = 0.0) -> np.ndarray:
        idx = self.col_index[key]
        col_values = self.data[:, idx]
        values = np.roll(col_values, shift)
        if shift > 0:
            for i in range(shift):
                values[i] = filler_value
        elif shift < 0:
            for i in range(shift, 0):
                values[i] = filler_value
        return values

    def round(self, decimals):
        self.data = np.round(self.data, decimals)

    def insert(self, row_dict: dict, index: int):
        """Insert a single row (dict of column: value) at the specified index."""
        new_row = np.full(self.data.shape[1], np.nan)
        for key, value in row_dict.items():
            # if key in self.col_index:
            new_row[self.col_index[key]] = value
        self.data = np.insert(self.data, index, new_row, axis=0)

    def update_row(self, index: int, row_dict: dict):
        for key, value in row_dict.items():
            if key in self.col_index:
                self.data[index, self.col_index[key]] = value

    def delete_row(self, index: int):
        self.data = np.delete(self.data, index, axis=0)

    def sort_by_column(self, column: str, ascending: bool = True):
        if column not in self.col_index:
            raise KeyError(f"Column {column} not found")
        col_data = self.data[:, self.col_index[column]]
        order = np.argsort(col_data)
        if not ascending:
            order = order[::-1]
        self.data = self.data[order]


def compare_problem_tables(pt1: ProblemTable, pt2: ProblemTable, atol: float = 1e-6) -> bool:
    """Compares two DataFrames element-wise and reports differences within an absolute tolerance."""
    if pt1.shape != pt2.shape:
        print(f"❌ Shape mismatch: {pt1.shape} vs {pt2.shape}")
        return False

    if list(pt1.columns) != list(pt2.columns):
        print("❌ Column mismatch:")
        print(f"pt1 columns: {pt1.columns}")
        print(f"pt2 columns: {pt2.columns}")
        return False

    mismatches = []
    for i in range(len(pt1)):
        for col in pt1.columns:
            v1, v2 = pt1.iloc[i][col], pt2.iloc[i][col]
            try:
                if pd.isna(v1) and pd.isna(v2):
                    continue
                if not np.isclose(v1, v2, atol=atol):
                    mismatches.append((i, col, v1, v2))
            except TypeError:
                if v1 != v2:
                    mismatches.append((i, col, v1, v2))

    if mismatches:
        print(f"⚠️ {len(mismatches)} mismatches found:")
        for i, col, v1, v2 in mismatches:
            print(f"Row {i}, Column '{col}': pt1={v1}, pt2={v2}")
        return False

    print("✅ All values match within tolerance.")
    return True
