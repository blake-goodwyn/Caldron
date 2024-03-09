from extract_and_clean import ing_clean
import pandas as pd
import ast
import re

# Example DataFrame
file = 'data/tart/processed-tart-recipe-2024-02-29-2229.csv'
#file = 'C:/Users/blake/Documents/GitHub/ebakery/data/tart/processed-tart-recipe-2024-02-29-2229.csv'
df = pd.read_csv(file)
file_new = 'data/tart/processed-tart-recipe-2024-02-29-2229-PROCESSED.csv'

# Specify the column to check for NaN
column_name = 'Processed Ingredients'

_DEBUG = False

def safely_convert_to_list(string_value):
    """
    Safely converts a string representation of a list into an actual list.
    Strips away outer quotation marks and tries to parse the remaining content as a list.
    """
    # Keep stripping quotation marks until we no longer have a string
    while isinstance(string_value, str):
        try:
            # Try to parse the string as a literal (list, dict, etc.)
            string_value = eval(string_value)
        except (ValueError, SyntaxError):
            # If it fails to parse, it's not a valid Python literal
            # So we consider it a final value (even if it's a string)
            break

    # If the result is a list, return it, otherwise, return the value inside a list
    return string_value if isinstance(string_value, list) else [string_value]

# Loop through each row and check for NaN in the specified column
def lint_dataset(df, _DEBUG=False):
    assert type(df) == pd.DataFrame, "Input must be a DataFrame"
    for index, row in df.iterrows():
        
        if pd.isna(row[column_name]):
            if _DEBUG:
                print("[Value Undefined] Cleaning row #", index+1, " | ", row['ID'])
            row[column_name] = ing_clean(row['Ingredients'])
            try:
                eval(row[column_name])
                assert type(eval(row[column_name]) == list)
                if _DEBUG:
                    print(row['ID'], " | PASS | ", row[column_name])
            except AssertionError as e:
                if _DEBUG:
                    print(e)
                row[column_name] = safely_convert_to_list(row[column_name])
            except Exception as e:
                if _DEBUG:
                    print(e)
                    print(row['ID'], " | FAIL | ", row[column_name])
        
        try:
            if _DEBUG:
                print("Evaluating row #", index+1, " | ", row['ID'])
            eval(row[column_name])
            assert type(eval(row[column_name]) == list)
            if _DEBUG:
                print(row['ID'], " | PASS | ", row[column_name])
        except AssertionError as e:
            if _DEBUG:
                print(e)
            row[column_name] = safely_convert_to_list(eval(row[column_name]))
            try:
                eval(row[column_name])
                assert type(eval(row[column_name]) == list)
                if _DEBUG:
                    print(row['ID'], " | PASS | ", row[column_name],"\n")
            except Exception as e:
                if _DEBUG:
                    print(e)
                    print(row['ID'], " | FAIL | ", row[column_name],"\n")
        except Exception as e:
            if _DEBUG:
                print(e)
                print("[Eval failed] Cleaning row #", index+1, " | ", row['ID'])
            row[column_name] = ing_clean(row['Ingredients'])
            try:
                eval(row[column_name])
                assert type(eval(row[column_name]) == list)
                if _DEBUG:
                    print(row['ID'], " | PASS | ", row[column_name],"\n")
            except Exception as e:
                if _DEBUG:
                    print(e)
                    print(row['ID'], " | FAIL | ", row[column_name],"\n")

    # Parse corrected dataframe
    c = 0
    for index, row in df.iterrows():
        #print("Evaluating row #", index+1, " | ", row['ID'])
        try:
            eval(row[column_name])
            assert type(eval(row[column_name])) == list
            if _DEBUG:
                print("Type: ", type(eval(row[column_name])))
            c+=1
            if _DEBUG:
                print("PASS | ", row['ID'],"\n")
        except Exception as e:
            if _DEBUG:
                print(e)
                print(row[column_name])
            print("FAIL | ", row['ID'])

    print("Percentage of rows readable: ", c/len(df)*100, "%")

    return df

def lint(file):
    assert type(file) == str, "Input must be a string"
    assert file.endswith('.csv'), "Input must be a CSV file"
    df = pd.read_csv(file)
    df = lint_dataset(df, _DEBUG=True)
    df.to_csv(file, index=False)

#lint_dataset(df)