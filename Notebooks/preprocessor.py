
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")


label_cols = ['Make', 'MaritalStatus']
freq_cols = ['PolicyType']
ordinal_cols = ['VehiclePrice', 'Days_Policy_Accident', 'PastNumberOfClaims', 
                'AgeOfVehicle', 'NumberOfSuppliments', 'AddressChange_Claim', 
                'NumberOfCars', 'DriverRating', 'Deductible']
binary_cols = ['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType']
scale_cols = ['Age', 'delay_weeks']


# -----------------------------
# Binary encoder
# -----------------------------
class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_map = {
            'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1, 'Urban': 1, 'Rural': 0, 
            'Policy Holder': 1, 'Third Party': 0, 'External': 0, 'Internal': 1
        }
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col in X_trans.columns:
            X_trans[col] = X_trans[col].map(self.binary_map)
        return X_trans

# -----------------------------
# Label encoding multiple colonnes
# -----------------------------


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_dict = {}
        self.classes_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            values = X[col].astype(str).tolist()

            # Ajouter une classe spéciale "unknown"
            unique_vals = list(set(values))
            unique_vals.append("__unknown__")

            le.fit(unique_vals)
            self.le_dict[col] = le
            self.classes_[col] = set(unique_vals)

        return self

    def transform(self, X):
        X_trans = X.copy()

        for col in X.columns:
            le = self.le_dict[col]

            # Remplacer les valeurs inconnues
            X_trans[col] = X_trans[col].astype(str).apply(
                lambda v: v if v in self.classes_[col] else "__unknown__"
            )

            X_trans[col] = le.transform(X_trans[col])

        return X_trans


# -----------------------------
# Frequency encoding
# -----------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, default_value=0):
        self.freq_map = {}
        self.default_value = default_value
        
    def fit(self, X, y=None):
        for col in X.columns:
            freq = X[col].value_counts() / len(X)
            self.freq_map[col] = freq.to_dict()
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            # map avec valeur par défaut si modalité inconnue
            X_encoded[col] = X_encoded[col].map(self.freq_map[col]).fillna(self.default_value)
        return X_encoded

# -----------------------------
# Ordinal mapping automatique
# -----------------------------


class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col, map_dict in self.mapping_dict.items():
            X_trans[col] = X_trans[col].map(map_dict)
        return X_trans