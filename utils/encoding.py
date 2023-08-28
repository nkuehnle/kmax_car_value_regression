from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing import Optional
from functools import reduce
import pandas as pd
import re
from typing import Dict, Sequence, List


class PandasEncoder:
    def __init__(self,
                 binary_labels,
                 multiclass_labels,
                 ordinal_labels,
                 numeric_labels
                 ):
        self.binary_labels = binary_labels
        self.multiclass_labels = multiclass_labels
        self.ordinal_labels = ordinal_labels
        self.numeric_labels = numeric_labels
        self.binary_encoder = None
        self.multiclass_encoder = None
        self.ordinal_encoder = None

    def append_to_labs(slf, lab: str, al: List[str]) -> List[str]:
        return [f"{lab}_{a}" for a in al]

    @property
    def aliases(self) -> Dict[str, Sequence[str]]:
        aliases = {}

        bin_cats = [c.tolist() for c in self.binary_encoder.categories_]
        bin_dict = zip(self.binary_labels, bin_cats)
        bin_aliases = {lab: [f"{lab}_{alias[1]}"] for lab, alias in bin_dict}

        mc_cats = [c.tolist() for c in self.multiclass_encoder.categories_]
        mc_dict = zip(self.multiclass_labels, mc_cats)
        mc_aliases = {lbl: self.append_to_labs(lbl, al) for lbl, al in mc_dict}

        ord_aliases = {lbl: [lbl] for lbl in self.ordinal_labels}

        num_aliases = {lbl: [lbl] for lbl in self.numeric_labels}

        aliases.update(bin_aliases)
        aliases.update(mc_aliases)
        aliases.update(ord_aliases)
        aliases.update(num_aliases)

        return aliases

    def transform_column_names(self, columns: Sequence[str]) -> Sequence[str]:
        aliases = self.aliases

        new_cols: List[str] = []
        for c in columns:
            new_cols.extend(aliases[c])

        return new_cols

    def fit(self, data: pd.DataFrame) -> "PandasEncoder":

        self.binary_encoder = OneHotEncoder().fit(data[self.binary_labels])

        self.multiclass_encoder = OneHotEncoder()
        self.multiclass_encoder.fit(data[self.multiclass_labels])
       
        ordinal_cats = [data[c].dtype._categories for c in self.ordinal_labels]
        self.ordinal_encoder = OrdinalEncoder(categories=ordinal_cats)
        self.ordinal_encoder.fit(data[self.ordinal_labels])

        return self

    def _binary_transform(self, binary_data: pd.DataFrame) -> pd.DataFrame:
        new_data = self.binary_encoder.transform(binary_data).todense()
        new_cols = zip(self.binary_labels, self.binary_encoder.categories_)
        new_cols = [f"{v}_{c[1]}" for v, c in new_cols]
        new_data = pd.DataFrame(new_data[:, 1::2], columns=new_cols)

        return new_data

    def _mc_transform(self, mc_data: pd.DataFrame) -> pd.DataFrame:
        encoder = self.multiclass_encoder
        new_data = encoder.transform(mc_data).todense()

        new_cols: List[str]= []
        for i, col in enumerate(mc_data.columns):
            _new_cols = [f"{col}_{cat}" for cat in encoder.categories_[i]]
            new_cols = new_cols + _new_cols
       
        new_data = pd.DataFrame(new_data, columns=new_cols)

        return new_data

    def _ordinal_transform(self, ordinal_data: pd.DataFrame) -> pd.DataFrame:
        new_data = self.ordinal_encoder.transform(ordinal_data)
        new_data = pd.DataFrame(new_data, columns=self.ordinal_labels)

        return new_data

    def _merge_help(self,
                    left: pd.DataFrame,
                    right: Optional[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(right, pd.DataFrame) and isinstance(left, pd.DataFrame):
            return pd.merge(left, right, left_index=True, right_index=True)
        else:
            return right

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        _data = data.copy()
        binary_data = self._binary_transform(_data[self.binary_labels])
        mc_data = self._mc_transform(_data[self.multiclass_labels])
        ord_data = self._ordinal_transform(_data[self.ordinal_labels])
        numeric_data = _data[self.numeric_labels]

        mfunc = lambda left, right: self._merge_help(left, right)
        _data = reduce(mfunc, [binary_data, mc_data, ord_data, numeric_data])

        return _data

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)


class RarityReEncoder:
    def __init__(self, min_makes: int = 10, min_models: int = 10):
        self.min_makes = min_makes
        self.min_models = min_models
        self.make_appr_mapper: Dict[str, str] = {}
        self.purch_make_mapper: Dict[str, str] = {}
        self.model_appr_mapper: Dict[str, str] = {}
        self.purch_model_mapper: Dict[str, str] = {}

    def get_make_mapper(self, df: pd.DataFrame, purchase: bool) -> Dict[str, str]:
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        purchase : bool
            _description_

        Returns
        -------
        Dict[str, str]
            _description_
        """
        col = "purchase_make" if purchase else "make_appraisal"
        counts = df[col].value_counts()
        rare_makes = counts[counts < self.min_makes].index
        return {m: "OTHER" for m in rare_makes}
        
    def multiple_str_match(self, series: pd.Series, strings: Sequence[str]) -> pd.Series:
        """
        Gets mask of rows in a series that contain one of any

        Parameters
        ----------
        series : pd.Series
            _description_
        strings : Sequence[str]
            _description_

        Returns
        -------
        pd.Series
            _description_
        """
        pattern = "|".join([re.escape(s) for s in strings])
        mask = series.str.contains(pattern, case=False)
        return mask

    def get_model_mapper(self, df: pd.DataFrame, purchase: bool) -> Dict[str, str]:
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        purchase : bool
            _description_

        Returns
        -------
        Dict[str, str]
            _description_
        """
        make_col = "purchase_make" if purchase else "make_appraisal"
        model_col = "purchase_model" if purchase else "model_appraisal"
        make_mapper = self.purch_make_mapper if purchase else self.make_appr_mapper

        low_count_makes = make_mapper.keys()
        low_count_models_by_make = df[df[make_col].isin(low_count_makes)][model_col].unique()

        model_mapper = {}
        for i, model in enumerate(low_count_models_by_make):
            model_mapper[model] = f"OTHER_{i}"

        counts = df.loc[~df[make_col].isin(make_mapper.keys()), model_col].value_counts()
        rare_models = counts[counts < self.min_models].index

        for model in rare_models:
            rare_model = model.split("_")[0] + "_RARE"
            model_mapper[model] = rare_model

        return model_mapper

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        new_data = df.copy()

        mappings = {
            "purchase_make": self.purch_make_mapper,
            "purchase_model": self.purch_model_mapper,
            "make_appraisal": self.make_appr_mapper,
            "model_appraisal": self.model_appr_mapper
            }

        for col, mapper in mappings.items():
            to_remap = new_data[col].isin(mapper.keys())
            new_data.loc[to_remap, col] = new_data.loc[to_remap, col].map(mapper)

        return new_data

    def fit(self, df: pd.DataFrame) -> "RarityReEncoder":
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        """
        self.make_appr_mapper = self.get_make_mapper(df, False)
        self.purch_make_mapper = self.get_make_mapper(df, True)
        self.model_appr_mapper = self.get_model_mapper(df, False)
        self.purch_model_mapper = self.get_model_mapper(df, True)

        return self

    def fit_transform(self, df: pd.DataFrame):
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.fit(df)
        return self.predict(df)


def create_pandas_encoder() -> PandasEncoder:
    binary_labels = [
        'purchase_trim_descrip',
        'trim_descrip_appraisal',
        'online_appraisal_flag'
        ]
    multiclass_labels = [
        'purchase_body',
        'body_appraisal',
        'purchase_make',
        'make_appraisal',
        'purchase_model',
        'model_appraisal',
        'purchase_color',
        'color_appraisal',
        'market'
        ]
    ordinal_labels = [
        'purchase_mileage',
        'mileage_appraisal',
        'purchase_price',
        'appraisal_offer'
        ]
    numeric_labels = [
        'purchase_model_year',
        'model_year_appraisal',
        'purchase_engine',
        'engine_appraisal',
        'purchase_mpg_city',
        'purchase_mpg_highway',
        'mpg_city_appraisal',
        'mpg_highway_appraisal',
        'purchase_cylinders',
        'cylinders_appraisal',
        'purchase_fuel_capacity',
        'fuel_capacity_appraisal',
        'purchase_horsepower',
        'horsepower_appraisal'
        ]

    encoder = PandasEncoder(binary_labels, multiclass_labels, ordinal_labels, numeric_labels)

    return encoder
