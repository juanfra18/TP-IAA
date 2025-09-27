import pandas as pd

class DataProcesser:
    
    dictionary: dict[str,dict[str,int]] = {
        "r8.2_who5_fct4": {
            "Poor": 1,
            "Good": 2,
            "Very good": 3,
            "Excellent": 4
        },
        "r8.2_who5_Iexcel": {
            "Other": 1,
            "Excellent": 2
        },
        "age_fct6": {
            "15-24": 1,
            "25-34": 2,
            "35-44": 3,
            "45-54": 4,
            "55-64": 5,
            "65+": 6
        },
        "gender_fct2": {
            "Male": 1,
            "Female": 2
        },
        "eth_fct4": {
            "European/Other": 1,
            "Maori": 2,
            "Pacific": 3,
            "Asian": 4
        },
        "education_qual": {
            "No formal qualification": 1,
            "High school": 2,
            "Certificate or diploma": 3,
            "Bachelor or above": 4
        },
        "r4_previnworkforce_fct3": {
            "Employed": 1,
            "Unemployed": 2,
            "Retired": 3
        },
        "r4.5_fct3": {
            "Not essential worker": 1,
            "Yes essential worker": 2
        },
        "r7.1_fct3": {
            "Never": 1,
            "Past": 2,
            "Current": 3
        },
        "income_band": {
            "$30,000 or less": 1,
            "$30,001– $70,000": 2,
            "$70,001 – $100,000": 3,
            "$100,001 – $150,000": 4,
            "$150,001 or more": 5,
            "Prefer not to say": 6
        },
        "r5.2_fct2": {
            "Poor or Fair": 1,
            "Good or better": 2
        },
        "r8.17_fct2": {
            "No": 1,
            "Yes past diagnosis": 2
        },
        "r5.10": {
            "No": 1,
            "Yes": 2
        },
        "r9.1_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r3.2_fct4": {
            "Live by myself": 1,
            "With one adult": 2,
            "With other adults": 3,
            "With children": 4
        },
        "r3.3_num": {},  # numeric count
        "r3.3_fct5": {
            "1": 1,
            "2": 2,
            "3-5": 3,
            "6-9": 4,
            "10+": 5
        },
        "r3.7_fct3": {
            "High": 1,
            "Medium": 2,
            "Low": 3
        },
        "r3.8": {
            "It has stayed the same": 1,
            "It has increased": 2,
            "It has decreased": 3
        },
        "r4_lesswork_fct2": {
            "Not less work": 1,
            "Less work": 2
        },
        "r4_lostwork_fct2": {
            "Not lost work": 1,
            "Lost work": 2
        },
        "r5.6_fct3": {
            "No": 1,
            "Suspected": 2,
            "Tested": 3,
            "Confirmed": 4
        },
        "r3.4": {
            "Not satisfied": 1,
            "Satisfied": 2,
            "Extremely satisfied": 3
        },
        "r3.10": {
            "Not well": 1,
            "Well": 2,
            "Very well": 3
        },
        "r3.11": {
            "None": 1,
            "A little": 2,
            "More than a little": 3
        },
        "r3.12": {
            "Less than 2 hours": 1,
            "Two plus hours": 2
        },
        "r8.16_1_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r8.16_2_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r8.16_3_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r8.16_6_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r8.16_4_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r12.1_11_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r12.1_13_fct2": {
            "No": 1,
            "Yes": 2
        },
        "r6.4_fct2": {
            "Low level": 1,
            "Hazardous level": 2
        },
        "r6.5_fct2": {
            "Low level": 1,
            "Hazardous level": 2
        },
        "r6_change_fct3": {
            "No change": 1,
            "Increase": 2,
            "Decrease": 3
        }
    }
    
    ignored_columns : list[str] = []

    MAX_SCORE : int = 25
    ignoring_threshold : float
    dataset : pd.DataFrame
    population : int
    score_key : str = "r8.2_who5_num"
    
    def __init__(self,dataset : pd.DataFrame, ignoring_threshold: float = 0.40) -> None:
        self.dataset = dataset
        self.ignoring_threshold = ignoring_threshold
        
        for col in dataset.columns:
            null_count = dataset[col].isnull().sum()
            null_percent = (null_count / len(dataset))
            if null_percent > self.ignoring_threshold:
                self.ignored_columns.append(col)
            dataset[col] = dataset[col].dropna()
            dataset[col] = dataset[dataset[col] != "."][col]
              
    
                
        self.population = dataset["postweight_unscaled"].sum()
                
    def process_dataset(self) -> pd.DataFrame:
        
        columns = [col for col in self.dictionary if col not in self.ignored_columns] + ["score", "probability"]
 
        df : pd.DataFrame = pd.DataFrame(columns=columns)
        
        for _, row in self.dataset.iterrows():
            entry : dict[str, str] = row.to_dict()
            vect , result  = self.process_entry(entry)
            if result:
                df.loc[len(df)] = vect
        
        return df

    
    def process_entry(self, entry: dict[str,str]) -> tuple[list,bool]:
        try:
            vect : list[int] = []
            
            for key in self.dictionary:
                if key in self.ignored_columns:
                    continue
                if self.dictionary[key] == {}:
                    vect.append(int(entry[key]))
                else:
                    vect.append(self.dictionary[key].get(entry[key], 0))
                
                
            score : float = float(entry[self.score_key])
            
            probability : float = int(entry["postweight_unscaled"]) / self.population
            
            return vect + [score/self.MAX_SCORE, probability], True
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
            return [], False
        
        
        
if __name__ == "__main__":
    df = pd.read_csv("datos/Resilience_CleanOnly_v1.csv", encoding="latin1")
    
    data_processer = DataProcesser(df)
    
    result_df = data_processer.process_dataset()
    
    result_df.to_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED.csv", index=False)