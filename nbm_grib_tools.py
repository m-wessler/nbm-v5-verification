"""
NBM v5.0 GRIB2 Indexing & Decoding Tools
========================================

A specialized toolkit for indexing, decoding, and unit-converting National Blend of Models (NBM)
GRIB2 files. This module handles specific NCEP ID collisions, hidden local tables (Fire Wx),
and standardizes units to US Imperial or Metric.

Usage:
    import nbm_grib_tools
    df = nbm_grib_tools.index_nbm5_grib("blend.t00z.qmd.f024.co.grib2")

Author:   Michael Wessler
Email:    michael.wessler@noaa.gov
Agency:   NOAA / National Weather Service
Created:  2026
"""

import struct
import re
import sys
from datetime import datetime
import pandas as pd
from eccodes import *

# ====================================================================================
#  CONSTANTS & LOOKUP TABLES
# ====================================================================================

UNIT_LOOKUP = {
    'RH': '%', 'RelHum': '%',
    'WSPD': 'm s**-1', 'WIND': 'm s**-1', 'Wind': 'm s**-1', 
    'GUST': 'm s**-1', 'Gust': 'm s**-1', 'WindGust': 'm s**-1',
    'T': 'K', 'Temp': 'K', 'MaxT': 'K', 'MinT': 'K', 'Dewpoint': 'K', 'Td': 'K',
    'SnowLevel': 'm', 'SnoLvl': 'm', 'CIG': 'm', 'Ceiling': 'm', 'VIS': 'm', 'Vis': 'm',
    'QPF': 'kg m**-2', 'Precip': 'kg m**-2', 'APCP': 'kg m**-2',
    'PoP12': '%', 'PoP': '%',
    'Haines': 'Index', 'LAL': 'Index',
    'MixHgt': 'm', 'TransWind': 'm s**-1', 'VentRate': 'm**2 s**-1',
    'ASNOW': 'm', 'SNOW': 'm', 'ICE': 'm',
    'CWASP': '%',
    'PTYPE': 'code'
}

ID_LOOKUP = {
    # --- MOISTURE ---
    (0, 1, 13):  {'name': 'Total Snowfall', 'units': 'm'},
    (0, 1, 37):  {'name': 'Convective Precip (Water)', 'units': 'kg m**-2'},
    (0, 1, 19):  {'name': 'Precipitation Type', 'units': 'code'},
    (0, 1, 226): {'name': 'Predominant Weather', 'units': 'Index'},
    (0, 1, 228): {'name': 'Total Snowfall (ASNOW)', 'units': 'm'},
    (0, 1, 229): {'name': 'Total Ice Accumulation', 'units': 'm'},
    (0, 1, 233): {'name': 'Snow Ratio', 'units': 'ratio'},

    # --- THERMAL ---
    (0, 0, 206): {'name': 'Wet Bulb Globe Temperature', 'units': 'K'},

    # --- MOMENTUM ---
    (0, 2, 22):  {'name': 'Wind Speed (Gust)', 'units': 'm s**-1'},
    (0, 2, 225): {'name': 'Transport Wind Speed', 'units': 'm s**-1'},
    (0, 2, 226): {'name': 'Transport Wind Direction', 'units': 'deg'},

    # --- MASS ---
    (0, 3, 192): {'name': 'MSLP (Eta Reduction)', 'units': 'Pa'},

    # --- CLOUD ---
    (0, 6, 1):   {'name': 'Total Cloud Cover', 'units': '%'},
    (0, 6, 11):  {'name': 'Ceiling', 'units': 'm'},
    (0, 6, 25):  {'name': 'Cloud Top Height', 'units': 'm'}, 

    # --- RADAR & AVIATION ---
    (0, 16, 3):   {'name': 'Echo Top', 'units': 'm'},
    (0, 16, 198): {'name': 'Max Reflectivity', 'units': 'dBZ'},
    (0, 19, 233): {'name': 'Icing Probability', 'units': '%'}, # ICPRB
    (0, 19, 238): {'name': 'Ellrod Index', 'units': 'Index'},  # ELLINX

    # --- PHYSICAL PROPERTIES ---
    (0, 19, 0):   {'name': 'Visibility', 'units': 'm'},
    (0, 19, 20):  {'name': 'Icing Potential (ICPRB)', 'units': 'Index'},
    (0, 19, 21):  {'name': 'Icing Severity (ICSEV)', 'units': 'Index'},
    (0, 19, 22):  {'name': 'Visibility (Cloud/Obscuration)', 'units': 'm'},
    (0, 19, 224): {'name': 'CWASP Index', 'units': '%'},
    
    # Fire Wx
    (0, 19, 235): {'name': 'Joint Fire Wx Prob', 'units': '%'},
    (0, 19, 236): {'name': 'Snow Level', 'units': 'm'},
    (0, 19, 237): {'name': 'Dry Thunderstorm Prob', 'units': '%'},
    (0, 19, 239): {'name': 'Snow Level', 'units': 'm'},

    # --- MARINE ---
    (10, 3, 204): {'name': 'Significant Wave Height', 'units': 'm'},
}

PTYPE_MAP = {1: 'Rain', 5: 'Snow', 3: 'Freezing Rain', 8: 'Ice Pellets'}

# ====================================================================================
#  INTERNAL HELPER FUNCTIONS
# ====================================================================================

def _get_datetime(gid, date_key, time_key):
    """Safe extraction of datetime objects from GRIB keys."""
    try:
        d = codes_get(gid, date_key)
        t = codes_get(gid, time_key)
        return datetime.strptime(f"{d}{t:04d}", "%Y%m%d%H%M")
    except: 
        return None

def _apply_scale(raw_value, scale_factor):
    """Applies decimal scaling factor (value * 10^-factor)."""
    if raw_value is None: return None
    if scale_factor is None: scale_factor = 0
    if raw_value > 1e19: return None 
    return raw_value * (10 ** -scale_factor)

def _get_derived_type(gid):
    """Decodes Derived Forecast Type (Table 4.7)."""
    try:
        code = codes_get(gid, 'derivedForecast')
        mapping = {
            0: "Mean", 1: "Weighted Mean", 2: "Std Dev", 4: "Spread",
            192: "Deterministic", # Unweighted Mode
            241: "Deterministic"  # Most Probable
        }
        return mapping.get(code, f"Derived Type {code}")
    except: return None

def _to_imperial(val, unit, short_name):
    """Robust conversion to US Imperial Units."""
    if val is None or unit is None: return val, unit
    
    # Temperature: Kelvin -> Fahrenheit
    if unit == 'K': 
        return (val - 273.15) * 1.8 + 32, 'F'
    
    # Speed: m/s -> MPH
    if unit == 'm s**-1': 
        mph = val * 2.23694
        return round(mph), 'mph'
    
    # Length / Accumulation: Meters -> Inches or Feet
    if unit == 'm':
        accum_vars = ['APCP', 'ASNOW', 'SNOD', 'FICEAC', 'TICE', 'SNOWLR', 'WEASD']
        if short_name in accum_vars: return val * 39.3701, 'in'
        else: return val * 3.28084, 'ft'

    # Precip Amount: kg/m^2 (mm) -> Inches
    if unit == 'kg m**-2' or unit == 'mm': return val * 0.0393701, 'in'
    if unit == 'cm': return val * 0.393701, 'in'
    
    return val, unit

def _format_val(val, unit):
    """Formats values cleanly based on unit type."""
    if val is None: return "NaN"
    if isinstance(val, (int, float)):
        if unit in ['%', 'mph', 'deg', 'Index']: return f"{int(round(val))}"
        return f"{val:.2f}"
    return str(val)

def _parse_local_text_full(text):
    """Parses hidden Section 2 ASCII text (e.g., 'RH_le_35_WSPD_ge_10')."""
    if not text: return None, []
    ops = {'le': '<=', 'lt': '<', 'ge': '>=', 'gt': '>', 'eq': '=', 'ne': '!='}
    pattern = re.compile(r"([A-Za-z0-9]+)_(le|lt|ge|gt|eq|ne)_([0-9\.]+)")
    matches = pattern.findall(text)
    if not matches: return text, []
    
    desc_parts, parsed_items = [], []
    for var, op, val in matches:
        readable_op = ops.get(op, op)
        desc_parts.append(f"{var} {readable_op} {val}")
        try: numeric_val = float(val)
        except: numeric_val = None
        parsed_items.append({'var': var, 'op': readable_op, 'val': numeric_val})
        
    full_readable = "Prob " + " & ".join(desc_parts)
    return full_readable, parsed_items

# ====================================================================================
#  PUBLIC API
# ====================================================================================

def index_nbm5_grib(filename, convert_imperial=True):
    data = []
    print(f"[nbm_grib_tools] Indexing {filename}...")

    with open(filename, 'rb') as f:
        count = 0
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None: break
            count += 1
            
            row = {
                'msg_id': count, 'grib_header': None, 'shortName': None, 'name': None, 
                'level': None, 'typeOfLevel': None, 'init_time': None, 'valid_time': None, 
                'f_hour': None, 'stepRange': None, 'stepType': None, 'param_type': 'Deterministic', 
                'percentile': None, 'threshold': None, 'threshold_condition': None, 'units': None,
                'threshold_joint': None, 'threshold_condition_joint': None, 'units_joint': None,
                'full_desc': None
            }

            try:
                # --- READ METADATA ---
                d = codes_get(gid, 'discipline')
                c = codes_get(gid, 'parameterCategory')
                n = codes_get(gid, 'parameterNumber')
                raw_short = codes_get(gid, 'shortName')
                
                row['level'] = codes_get(gid, 'level')
                row['typeOfLevel'] = codes_get(gid, 'typeOfLevel')
                row['stepRange'] = codes_get(gid, 'stepRange')
                row['stepType'] = codes_get(gid, 'stepType')
                
                try: pdt = codes_get(gid, 'productDefinitionTemplateNumber')
                except: pdt = 0
                try: grib_units = codes_get(gid, 'units')
                except: grib_units = '-'

                row['init_time'] = _get_datetime(gid, 'dataDate', 'dataTime')
                row['valid_time'] = _get_datetime(gid, 'validityDate', 'validityTime')
                row['f_hour'] = codes_get(gid, 'startStep')
                time_str = row['init_time'].strftime("%Y%m%d%H") if row['init_time'] else "T-UNK"

                # --- 1. RESOLVE NAME/UNITS (COLLISION HANDLING) ---
                if (d, c, n) == (0, 19, 239):
                    is_prob = (pdt in [5, 9])
                    is_derived = (pdt in [2, 12])
                    if is_prob or is_derived: 
                        row['name'], row['units'], row['shortName'] = 'CWASP Index', '%', 'CWASP'
                    elif row['typeOfLevel'] == 'meanSea': 
                        row['name'], row['units'], row['shortName'] = 'Snow Level', 'm', 'SNOWLVL'
                    else:
                        if grib_units == 'm': row['name'], row['units'], row['shortName'] = 'Snow Level', 'm', 'SNOWLVL'
                        else: row['name'], row['units'], row['shortName'] = 'CWASP Index', '%', 'CWASP'
                
                elif (d, c, n) == (0, 1, 29):
                    is_prob = (pdt in [5, 9])
                    is_accum = (row['stepType'] == 'accum')
                    if is_prob or is_accum or raw_short == 'ASNOW':
                        row['name'], row['units'], row['shortName'] = 'Total Snowfall', 'm', 'ASNOW'
                    else:
                        row['name'], row['units'], row['shortName'] = 'Snow Ratio', 'ratio', 'SNOWLR'

                elif (d, c, n) in ID_LOOKUP:
                    entry = ID_LOOKUP[(d, c, n)]
                    row['name'], row['units'], row['shortName'] = entry['name'], entry['units'], raw_short
                elif codes_get(gid, 'name') != 'unknown':
                    row['name'], row['units'], row['shortName'] = codes_get(gid, 'name'), grib_units, raw_short
                else:
                    row['name'], row['units'], row['shortName'] = f"Unknown (D{d}-C{c}-N{n})", grib_units, f"unk_{d}_{c}_{n}"

                # --- 2. HIDDEN TEXT CHECK ---
                local_text = None
                msg_bytes = codes_get_message(gid)
                offset = 16
                sec1_len = struct.unpack('>I', msg_bytes[offset:offset+4])[0]
                offset += sec1_len
                sec2_len = struct.unpack('>I', msg_bytes[offset:offset+4])[0]
                sec2_num = msg_bytes[offset+4]
                if sec2_num == 2:
                    data_start = offset + 6
                    data_end = offset + sec2_len
                    try:
                        raw = msg_bytes[data_start:data_end]
                        decoded = raw.replace(b'\x00', b'').decode('ascii').strip()
                        if len(decoded) > 1 and any(ch.isalpha() for ch in decoded): local_text = decoded
                    except: pass

                # --- 3. PARSING ---
                if local_text:
                    readable, items = _parse_local_text_full(local_text)
                    if items:
                        row['param_type'] = 'Probability'
                        row['threshold'] = items[0]['val']
                        row['threshold_condition'] = items[0]['op']
                        row['units'] = UNIT_LOOKUP.get(items[0]['var'], 'Unknown')
                        var_names = [items[0]['var']]
                        
                        if convert_imperial:
                            row['threshold'], row['units'] = _to_imperial(row['threshold'], row['units'], items[0]['var'])

                        if len(items) > 1:
                            row['param_type'] = 'Joint Probability'
                            row['threshold_joint'] = items[1]['val']
                            row['threshold_condition_joint'] = items[1]['op']
                            row['units_joint'] = UNIT_LOOKUP.get(items[1]['var'], 'Unknown')
                            var_names.append(items[1]['var'])
                            if convert_imperial:
                                row['threshold_joint'], row['units_joint'] = _to_imperial(row['threshold_joint'], row['units_joint'], items[1]['var'])

                        row['name'] = f"Prob ({' & '.join(var_names)})"
                        row['shortName'] = "JFWPRB" if len(items) > 1 else raw_short
                        
                        desc_parts = []
                        if row['threshold'] is not None: 
                            desc_parts.append(f"{var_names[0]} {row['threshold_condition']} {_format_val(row['threshold'], row['units'])}")
                        if len(items) > 1 and row['threshold_joint'] is not None:
                            desc_parts.append(f"{var_names[1]} {row['threshold_condition_joint']} {_format_val(row['threshold_joint'], row['units_joint'])}")
                        row['full_desc'] = "Prob " + " & ".join(desc_parts) + f" ({row['units']})"
                    else:
                        row['name'] = f"{local_text} [ASCII]"
                        row['full_desc'] = row['name']
                        row['param_type'] = 'Local'
                else:
                    try:
                        if pdt in [6, 10]:
                            row['param_type'] = 'Percentile'
                            val = codes_get(gid, 'percentileValue')
                            row['percentile'] = val if val != 255 else None
                            if convert_imperial: _, row['units'] = _to_imperial(0, row['units'], row['shortName'])
                            row['full_desc'] = f"{row['name']} ({row['percentile']}th Pct)"

                        elif pdt in [5, 9]:
                            row['param_type'] = 'Probability'
                            prob_type = codes_get(gid, 'probabilityType')
                            if prob_type == 0: raw_val, scale, cond = codes_get(gid, 'scaledValueOfLowerLimit'), codes_get(gid, 'scaleFactorOfLowerLimit'), '<'
                            elif prob_type == 1: raw_val, scale, cond = codes_get(gid, 'scaledValueOfUpperLimit'), codes_get(gid, 'scaleFactorOfUpperLimit'), '>'
                            elif prob_type == 2:
                                lower = _apply_scale(codes_get(gid, 'scaledValueOfLowerLimit'), codes_get(gid, 'scaleFactorOfLowerLimit'))
                                if 'Precipitation Type' in row['name'] and int(lower) in PTYPE_MAP:
                                    row['threshold'], row['threshold_condition'] = int(lower), '=='
                                    row['full_desc'] = f"Prob Type = {PTYPE_MAP[int(lower)]}"
                                    raw_val, scale, cond = None, 0, '=='
                                else:
                                    raw_val, scale, cond = codes_get(gid, 'scaledValueOfLowerLimit'), codes_get(gid, 'scaleFactorOfLowerLimit'), 'between'
                            else: raw_val, scale, cond = codes_get(gid, 'scaledValueOfLowerLimit'), codes_get(gid, 'scaleFactorOfLowerLimit'), '?'

                            if prob_type != 2:
                                val = _apply_scale(raw_val, scale)
                                if convert_imperial and 'Type' not in row['name']: val, row['units'] = _to_imperial(val, row['units'], row['shortName'])
                                row['threshold'], row['threshold_condition'] = val, cond
                                row['full_desc'] = f"Prob {row['name']} {cond} {_format_val(val, row['units'])} {row['units']}"

                        elif pdt in [2, 12]:
                            d_type = _get_derived_type(gid)
                            row['param_type'] = d_type if d_type else 'Deterministic'
                            if convert_imperial: _, row['units'] = _to_imperial(0, row['units'], row['shortName'])
                            row['full_desc'] = f"{row['name']} ({row['param_type']})"
                        else:
                            if convert_imperial: _, row['units'] = _to_imperial(0, row['units'], row['shortName'])
                            row['full_desc'] = row['name']
                    except: row['full_desc'] = f"{row['name']} (Error)"

                if row['stepType'] and row['stepType'] != 'instant':
                    row['full_desc'] += f" [{row['stepRange']} hr {row['stepType']}]"

                # --- 4. CLEANUP FOR "UNKNOWN" ---
                # Recover Names/ShortNames if ID lookup missed but we have clues
                if 'Unknown' in row['name'] or 'unknown' in row['shortName']:
                     if local_text and 'Cloud' in local_text:
                         row['name'] = local_text
                         row['shortName'] = local_text.upper()[:8]
                     
                     if row['name'] == 'Ceiling': row['shortName'] = 'CEIL'
                     elif row['name'] == 'Total Cloud Cover': row['shortName'] = 'TCC'
                     elif row['name'] == 'Icing Severity': row['shortName'] = 'ICSEV'
                     elif row['name'] == 'Predominant Weather': row['shortName'] = 'WX'
                     elif row['name'] == 'Echo Top': row['shortName'] = 'ECHOTOP'
                     elif row['name'] == 'Ellrod Index': row['shortName'] = 'ELLROD'
                     elif row['name'] == 'Dry Thunderstorm Prob': row['shortName'] = 'DRYTS'
                     elif row['name'] == 'Transport Wind Speed': row['shortName'] = 'TRWSPD'
                     elif row['name'] == 'Transport Wind Direction': row['shortName'] = 'TRWDIR'

                # --- 5. CLEAN HEADER GENERATION (AFTER FIXES) ---
                # Regenerate header so it reflects the corrected shortName/Name
                row['grib_header'] = f"{count}:{row['shortName']}:{row['typeOfLevel']}={row['level']}:{row['stepRange']}hr {row['stepType']}:d={time_str}"

                data.append(row)
            except Exception: pass
            finally: codes_release(gid)

    df = pd.DataFrame(data)
    cols_order = ['msg_id', 'grib_header', 'shortName', 'name', 'init_time', 'f_hour', 'valid_time', 'stepRange', 'stepType', 'level', 'typeOfLevel', 'param_type', 'percentile', 'threshold_condition', 'threshold', 'units', 'threshold_condition_joint', 'threshold_joint', 'units_joint', 'full_desc']
    return df[[c for c in cols_order if c in df.columns]]

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    if len(sys.argv) < 2: print("Usage: python nbm_grib_tools.py <grib_file>")
    else:
        try:
            df = index_nbm5_grib(sys.argv[1], convert_imperial=True)
            print(f"\n[+] Success: Indexed {len(df)} messages.")
            # Print sample of fixed unknowns
            mask = df['name'].str.contains('Unknown', na=False)
            if mask.any():
                print("\n[!] Remaining Unknowns (if any):")
                print(df[mask][['msg_id', 'shortName', 'name', 'full_desc']].to_string(index=False))
            else:
                print("\n[+] All parameters successfully identified.")
                print(df[['msg_id', 'grib_header', 'name', 'full_desc']].to_string(index=False))

        except Exception as e: print(f"[!] Error: {e}")