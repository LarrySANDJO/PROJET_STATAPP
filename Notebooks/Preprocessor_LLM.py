import pandas as pd

class PreprocessLLM:
    def __init__(self, label_cols, freq_cols, ordinal_cols, binary_cols, scale_cols):
        self.label_cols = label_cols
        self.freq_cols = freq_cols
        self.ordinal_cols = ordinal_cols
        self.binary_cols = binary_cols
        self.scale_cols = scale_cols
    
    def verbalize_row(self, row):
        lines = []
        if "Age" in self.scale_cols:
            lines.append(f"Âge : {row['Age']} ans")
        if "delay_weeks" in self.scale_cols:
            lines.append(f"Délai avant déclaration : {row['delay_weeks']} semaines")
        
        binary_map = {
            "Sex": "Sexe", 
            "AccidentArea": "Zone de l'accident",
            "Fault": "Responsabilité", 
            "PoliceReportFiled": "Rapport de police",
            "WitnessPresent": "Présence de témoins", 
            "AgentType": "Type d'agent"
        }
        for col, label in binary_map.items():
            if col in self.binary_cols:
                lines.append(f"{label} : {row[col]}")
        
        for col in self.ordinal_cols:
            lines.append(f"{col} : {row[col]}")
        for col in self.freq_cols:
            lines.append(f"{col} : {row[col]}")
        for col in self.label_cols:
            lines.append(f"{col} : {row[col]}")
        
        return "\n".join(lines)
    
    def transform(self, df, id_col=None):
        outputs = []
        for idx, row in df.iterrows():
            obs_id = row[id_col] if id_col else idx
            text = self.verbalize_row(row)
            outputs.append({"id": obs_id, "text": text})
        return outputs