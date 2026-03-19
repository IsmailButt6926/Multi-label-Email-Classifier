# Methods related to data loading and all pre-processing steps will go here
import os
import re
import numpy as np
import pandas as pd
from Config import Config


def get_input_data():
    """Load input data from all CSV files specified in Config and concatenate them."""
    frames = []
    for filename in Config.DATA_FILES:
        filepath = os.path.join(Config.DATA_DIR, filename)
        df = pd.read_csv(filepath)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Rename type columns to internal names (y1, y2, y3, y4)
    df.rename(columns=Config.COLUMN_RENAME, inplace=True)

    # Strip whitespace from type columns
    for col in ['y1', 'y2', 'y3', 'y4']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Convert 'nan' strings back to actual NaN
            df[col] = df[col].replace('nan', np.nan)

    # Drop any extra unnamed columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    return df


def de_duplication(df):
    """Remove duplicate entries based on Interaction content."""
    df = df.drop_duplicates(subset=[Config.INTERACTION_CONTENT], keep='first')
    df = df.reset_index(drop=True)
    return df


def noise_remover(df):
    """Clean noise from text fields: HTML entities, special characters, extra whitespace."""
    def clean_text(text):
        if not isinstance(text, str):
            return text
        # Decode HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove phone-like patterns masked as *****(PHONE)
        text = re.sub(r'\*+\([A-Z]+\)', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].apply(clean_text)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].apply(clean_text)
    return df


def translate_to_en(texts):
    """Translate a list of texts to English. Falls back to original text if translation unavailable."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        translated = []
        for text in texts:
            try:
                if isinstance(text, str) and len(text.strip()) > 0:
                    result = translator.translate(text[:5000])  # API character limit
                    translated.append(result if result else text)
                else:
                    translated.append(text)
            except Exception:
                translated.append(text)
        return translated
    except ImportError:
        # deep_translator not installed; return original texts
        print("Note: deep_translator not installed. Skipping translation.")
        return texts


def create_chained_columns(df):
    """Create chained target columns for multi-label classification (Design Choice 1).

    Creates:
        y_chain_1 = y2
        y_chain_2 = y2 + y3  (concatenated string)
        y_chain_3 = y2 + y3 + y4  (concatenated string)
    """
    sep = Config.CHAIN_SEPARATOR

    for chain_name, cols in Config.CHAINED_TARGETS.items():
        col_name = 'y_' + chain_name
        # Start with the first column
        df[col_name] = df[cols[0]].astype(str)
        # Concatenate subsequent columns
        for col in cols[1:]:
            df[col_name] = df[col_name] + sep + df[col].astype(str)

        # Mark rows with NaN in any contributing column as NaN in the chained column
        for col in cols:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == '') | (df[col].astype(str) == 'nan')
            df.loc[mask, col_name] = np.nan

    return df
