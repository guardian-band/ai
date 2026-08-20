import re
import pandas as pd
import numpy as np

MEDDRA_SOC_MAP = {
    'Cardiovascular & Vascular': [
        'cardiac', 'coronary', 'arrhythmia', 'hypertension', 'hypotension', 'myocardial',
        'tachycardia', 'bradycardia', 'angina', 'vascular', 'aneurysm', 'thrombosis',
        'heart', 'arter', 'vaso', 'atrial', 'ventricular', 'circulatory', 'palpitation'
    ],
    'Gastrointestinal': [
        'abdominal', 'gastric', 'ulcer', 'nausea', 'vomit', 'diarrhea', 'constipation',
        'dyspepsia', 'colitis', 'esophag', 'intestin', 'pancreat', 'bowel', 'rectal',
        'gastro', 'stomach', 'oral', 'mouth', 'stomatitis', 'gerd', 'flatulence', 'dysphagia'
    ],
    'Nervous System & Neurological': [
        'headache', 'dizzy', 'dizziness', 'convulsion', 'seizure', 'tremor', 'neuropathy',
        'coma', 'encephalopathy', 'ataxia', 'somnolence', 'stroke', 'amnesia', 'neural',
        'brain', 'paresthesia', 'sedation', 'syncope', 'epilepsy', 'paralysis', 'migraine'
    ],
    'Hepatobiliary (Liver)': [
        'hepatic', 'liver', 'hepatitis', 'jaundice', 'bilirubin', 'transaminase',
        'cirrhosis', 'gallbladder', 'bile', 'hepatotoxicity', 'alt', 'ast', 'cholecyst'
    ],
    'Renal & Urinary (Kidney)': [
        'renal', 'kidney', 'nephritis', 'proteinuria', 'oliguria', 'hematuria',
        'creatinine', 'dialysis', 'bladder', 'urinary', 'ureter', 'micturition', 'nephro'
    ],
    'Respiratory & Thoracic (Lungs)': [
        'pulmonary', 'asthma', 'cough', 'dyspnea', 'bronchitis', 'pneumonia',
        'bronchospasm', 'lung', 'respiratory', 'pharyngitis', 'rhinitis', 'pleural',
        'sinusitis', 'atelectasis'
    ],
    'Hematologic & Lymphatic (Blood)': [
        'anemia', 'leukopenia', 'thrombocytopenia', 'hemorrhage', 'bleeding',
        'neutropenia', 'clotting', 'hemoglobin', 'lymph', 'pancytopenia', 'coagul',
        'purpura', 'ecchymosis', 'hematoma', 'blood', 'platelet', 'eosinophilia'
    ],
    'Dermatologic (Skin & Hair)': [
        'rash', 'pruritus', 'erythema', 'dermatitis', 'alopecia', 'urticaria',
        'eczema', 'skin', 'ulceration', 'bullous', 'dermal', 'photosensitivity',
        'sweat', 'psoriasis', 'acne', 'pigmentation', 'dry skin'
    ],
    'Psychiatric & Behavioral': [
        'depression', 'anxiety', 'insomnia', 'hallucination', 'agitation',
        'psychosis', 'confusion', 'delirium', 'bipolar', 'suicid', 'nervousness',
        'panic', 'nightmare', 'mood', 'behavior', 'personality', 'restlessness'
    ],
    'Metabolism & Endocrine (Hormones)': [
        'hyperglycemia', 'hypoglycemia', 'diabetes', 'hyponatremia', 'hyperkalemia',
        'hypokalemia', 'acidosis', 'weight', 'thyroid', 'endocrine', 'calcium',
        'lipid', 'cholesterol', 'gout', 'anorexia', 'metabol', 'hyperuricemia'
    ],
    'Musculoskeletal & Connective Tissue': [
        'myalgia', 'arthralgia', 'arthritis', 'rhabdomyolysis', 'muscle',
        'fracture', 'osteoporosis', 'spasm', 'tendon', 'bone', 'joint', 'cramp',
        'myopathy', 'back pain', 'stiffness'
    ],
    'Immune System & Allergy': [
        'anaphylaxis', 'allergy', 'allergic', 'hypersensitivity', 'autoimmune',
        'angioedema', 'shock', 'lupus', 'graft', 'immune', 'serum sickness'
    ],
    'Ophthalmic & Otic (Eye & Ear)': [
        'visual', 'retinopathy', 'glaucoma', 'cataract', 'conjunctivitis',
        'blindness', 'vision', 'optic', 'eye', 'ocular', 'tinnitus', 'ear',
        'deafness', 'hearing', 'vertigo', 'blurred'
    ],
    'Infections & Infestations': [
        'sepsis', 'infection', 'bacteremia', 'fungal', 'viral', 'abscess',
        'meningitis', 'septic', 'cellulitis', 'candidiasis', 'herpes', 'influenza'
    ],
    'General Disorders & Systemic': [
        'fever', 'fatigue', 'asthenia', 'edema', 'pain', 'malaise', 'chills',
        'death', 'injection site', 'chest pain', 'weakness', 'pyrexia', 'edematous', 'feeling abnormal'
    ]
}

def map_side_effect_to_soc(name):
    name_l = str(name).lower()
    for soc, kws in MEDDRA_SOC_MAP.items():
        for kw in kws:
            if re.search(r'\b' + re.escape(kw), name_l) or kw in name_l:
                return soc
    return 'General Disorders & Systemic'

def build_meddra_hierarchical_mapping(side_effects_raw_path):
    df = pd.read_csv(side_effects_raw_path)
    cui_to_name = dict(zip(df['umls_cui_from_meddra'], df['side_effect_name']))
    cui_to_soc = {cui: map_side_effect_to_soc(name) for cui, name in cui_to_name.items()}
    soc_categories = list(MEDDRA_SOC_MAP.keys())
    return cui_to_soc, soc_categories
