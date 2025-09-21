
from argopy import gdacfs
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

def safe_extract_scalar(data_array):
    """Safely extract scalar values from xarray DataArray"""
    try:
        if data_array is None:
            return None
        
        value = data_array.values
        
        # Handle different data types
        if hasattr(value, 'item'):
            value = value.item()
        
        # Convert bytes to string and clean
        if isinstance(value, bytes):
            value = value.decode('utf-8').strip()
        elif isinstance(value, str):
            value = value.strip()
        elif isinstance(value, (np.ndarray, list)) and len(value) > 0:
            if isinstance(value[0], bytes):
                # Join byte arrays and decode
                value = ''.join([b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in value]).strip()
            elif len(value) == 1:
                value = value[0]
        
        # Handle NaN and empty values
        if pd.isna(value) or (isinstance(value, str) and (value == '' or value.lower() == 'nan')):
            return None
            
        return value
    except Exception:
        return None

def safe_extract_array_value(data_array, indices):
    """Safely extract single value from multidimensional array"""
    try:
        if data_array is None:
            return None
            
        value = data_array.values[indices]
        
        # Handle scalar results
        if np.isscalar(value):
            if isinstance(value, bytes):
                decoded = value.decode('utf-8').strip()
                return decoded if decoded else None
            elif pd.isna(value):
                return None
            else:
                return float(value) if isinstance(value, (np.floating, float)) else value
        
        return value
    except Exception:
        return None

def extract_float_metadata(fs, float_id, dac="coriolis"):
    """Extract float metadata from meta.nc file"""
    print(f"  📋 Extracting metadata for float {float_id}...")
    
    meta_file_path = f"dac/{dac}/{float_id}/{float_id}_meta.nc"
    
    try:
        with fs.open_dataset(meta_file_path) as ds_meta:
            # Core metadata fields
            metadata_fields = [
                'PLATFORM_NUMBER', 'PLATFORM_TYPE', 'PLATFORM_MAKER', 'FLOAT_SERIAL_NO',
                'PROJECT_NAME', 'PI_NAME', 'LAUNCH_DATE', 'LAUNCH_LATITUDE', 'LAUNCH_LONGITUDE',
                'START_DATE', 'END_MISSION_DATE', 'BATTERY_TYPE', 'FIRMWARE_VERSION',
                'DEPLOYMENT_PLATFORM', 'DEPLOYMENT_CRUISE_ID', 'FLOAT_OWNER', 
                'OPERATING_INSTITUTION', 'DATA_CENTRE', 'WMO_INST_TYPE'
            ]
            
            float_metadata = {'FLOAT_ID': float_id}
            
            for field in metadata_fields:
                if field in ds_meta.variables:
                    value = safe_extract_scalar(ds_meta[field])
                    float_metadata[field] = value
                else:
                    float_metadata[field] = None
        
        print(f"    ✅ Metadata extracted")
        return float_metadata
        
    except Exception as e:
        print(f"    ❌ Error extracting metadata: {e}")
        # Return empty metadata record so processing continues
        return {'FLOAT_ID': float_id, **{field: None for field in metadata_fields if 'field' in locals()}}

def extract_profile_and_measurement_data(fs, float_id, dac="coriolis", max_profiles=10):
    """Extract both profile info and measurements from prof.nc file"""
    print(f"  🌊 Extracting data from prof.nc for float {float_id}...")
    
    prof_file_path = f"dac/{dac}/{float_id}/{float_id}_prof.nc"
    
    try:
        with fs.open_dataset(prof_file_path) as ds_prof:
            n_profiles = ds_prof.dims.get('N_PROF', 0)
            n_levels = ds_prof.dims.get('N_LEVELS', 0)
            
            # Limit profiles to process
            profiles_to_process = min(n_profiles, max_profiles)
            print(f"    📊 Processing {profiles_to_process}/{n_profiles} profiles, {n_levels} levels each")
            
            profile_data = []
            measurements = []
            
            # Profile-level variables (1D arrays indexed by N_PROF)
            profile_vars = {
                'CYCLE_NUMBER': 'cycle_number', 
                'JULD': 'date_time', 
                'LATITUDE': 'latitude', 
                'LONGITUDE': 'longitude',
                'POSITION_QC': 'position_qc', 
                'DIRECTION': 'direction', 
                'DATA_MODE': 'data_mode',
                'PROFILE_PRES_QC': 'profile_pres_qc', 
                'PROFILE_TEMP_QC': 'profile_temp_qc', 
                'PROFILE_PSAL_QC': 'profile_psal_qc'
            }
            
            # Measurement variables (2D arrays indexed by N_PROF, N_LEVELS)
            measurement_vars = {
                'PRES': 'pressure', 
                'TEMP': 'temperature', 
                'PSAL': 'salinity',
                'PRES_QC': 'pressure_qc', 
                'TEMP_QC': 'temperature_qc', 
                'PSAL_QC': 'salinity_qc',
                'PRES_ADJUSTED': 'pressure_adjusted', 
                'TEMP_ADJUSTED': 'temperature_adjusted', 
                'PSAL_ADJUSTED': 'salinity_adjusted',
                'PRES_ADJUSTED_QC': 'pressure_adjusted_qc',
                'TEMP_ADJUSTED_QC': 'temperature_adjusted_qc', 
                'PSAL_ADJUSTED_QC': 'salinity_adjusted_qc'
            }
            
            # Add DOXY variables if they exist
            if 'DOXY' in ds_prof.variables:
                measurement_vars.update({
                    'DOXY': 'oxygen',
                    'DOXY_QC': 'oxygen_qc', 
                    'DOXY_ADJUSTED': 'oxygen_adjusted',
                    'DOXY_ADJUSTED_QC': 'oxygen_adjusted_qc'
                })
                profile_vars['PROFILE_DOXY_QC'] = 'profile_oxygen_qc'
            
            # Process each profile
            for prof_idx in range(profiles_to_process):
                # Extract profile-level information
                profile_info = {
                    'FLOAT_ID': float_id,
                    'PROFILE_NUMBER': prof_idx + 1
                }
                
                for var_name, col_name in profile_vars.items():
                    if var_name in ds_prof.variables:
                        value = safe_extract_array_value(ds_prof[var_name], prof_idx)
                        profile_info[var_name] = value
                    else:
                        profile_info[var_name] = None
                
                profile_data.append(profile_info)
                
                # Extract measurements for this profile
                profile_measurements = 0
                for level_idx in range(n_levels):
                    # Check if this level has pressure data (skip if not)
                    if 'PRES' in ds_prof.variables:
                        pres_val = safe_extract_array_value(ds_prof['PRES'], (prof_idx, level_idx))
                        if pres_val is None or pd.isna(pres_val):
                            continue  # Skip levels without pressure data
                    else:
                        continue
                    
                    # Create measurement record
                    measurement_row = {
                        'FLOAT_ID': float_id,
                        'PROFILE_NUMBER': prof_idx + 1,
                        'LEVEL': level_idx + 1,
                        'PRES': pres_val
                    }
                    
                    # Extract other measurement variables
                    for var_name, col_name in measurement_vars.items():
                        if var_name == 'PRES':  # Already handled above
                            continue
                            
                        if var_name in ds_prof.variables:
                            value = safe_extract_array_value(ds_prof[var_name], (prof_idx, level_idx))
                            measurement_row[var_name] = value
                        else:
                            measurement_row[var_name] = None
                    
                    measurements.append(measurement_row)
                    profile_measurements += 1
                
                print(f"    📈 Profile {prof_idx + 1}: {profile_measurements} measurements")
        
        print(f"    ✅ Extracted {len(profile_data)} profiles, {len(measurements)} measurements")
        return profile_data, measurements
        
    except Exception as e:
        print(f"    ❌ Error extracting data: {e}")
        return [], []

def check_float_exists(fs, float_id, dac="coriolis"):
    """Check if float files exist"""
    files_to_check = [
        f"dac/{dac}/{float_id}/{float_id}_meta.nc",
        f"dac/{dac}/{float_id}/{float_id}_prof.nc"
    ]
    
    existing_files = []
    for file_path in files_to_check:
        try:
            fs.info(file_path)
            existing_files.append(file_path.split('/')[-1])
        except:
            pass
    
    return len(existing_files) > 0, existing_files

def process_multiple_floats(float_ids, dac="coriolis", max_profiles=10):
    """Process multiple floats and extract all data"""
    print(f"🚀 ARGO DATA EXTRACTION STARTED")
    print(f"📊 Target: {len(float_ids)} floats, up to {max_profiles} profiles each")
    print("=" * 80)
    
    # Initialize file system
    fs = gdacfs("https://data-argo.ifremer.fr")
    
    # Storage for all data
    all_float_metadata = []
    all_profile_data = []
    all_measurements = []
    
    # Process each float
    for i, float_id in enumerate(float_ids, 1):
        print(f"\n🌊 PROCESSING FLOAT {i}/{len(float_ids)}: {float_id}")
        print("-" * 60)
        
        # Check if float exists
        exists, available_files = check_float_exists(fs, float_id, dac)
        if not exists:
            print(f"  ❌ Float {float_id} not found or no accessible files")
            continue
        
        print(f"  📁 Available files: {', '.join(available_files)}")
        
        try:
            # Extract float metadata
            float_metadata = extract_float_metadata(fs, float_id, dac)
            all_float_metadata.append(float_metadata)
            
            # Extract profile and measurement data
            profile_data, measurements = extract_profile_and_measurement_data(fs, float_id, dac, max_profiles)
            all_profile_data.extend(profile_data)
            all_measurements.extend(measurements)
            
            print(f"  ✅ Float {float_id} complete: {len(profile_data)} profiles, {len(measurements)} measurements")
            
        except Exception as e:
            print(f"  ❌ Error processing float {float_id}: {e}")
            continue
    
    return all_float_metadata, all_profile_data, all_measurements

def save_to_csv_files(all_float_metadata, all_profile_data, all_measurements):
    """Save extracted data to well-formatted CSV files"""
    print(f"\n💾 SAVING DATA TO CSV FILES")
    print("=" * 80)
    
    files_created = []
    
    # Save FLOAT.csv
    if all_float_metadata:
        df_floats = pd.DataFrame(all_float_metadata)
        df_floats.to_csv("FLOAT.csv", index=False)
        files_created.append(f"FLOAT.csv ({len(df_floats)} floats)")
        print(f"✅ FLOAT.csv saved: {len(df_floats)} floats")
        
        # Show sample
        print("   Sample data:")
        sample_cols = ['FLOAT_ID', 'PLATFORM_NUMBER', 'PROJECT_NAME', 'LAUNCH_LATITUDE', 'LAUNCH_LONGITUDE']
        available_cols = [col for col in sample_cols if col in df_floats.columns]
        if available_cols:
            print(df_floats[available_cols].head(3).to_string(index=False))
    
    # Save PROFILES.csv  
    if all_profile_data:
        df_profiles = pd.DataFrame(all_profile_data)
        df_profiles.to_csv("PROFILES.csv", index=False)
        files_created.append(f"PROFILES.csv ({len(df_profiles)} profiles)")
        print(f"\n✅ PROFILES.csv saved: {len(df_profiles)} profiles")
        
        # Show sample
        print("   Sample data:")
        sample_cols = ['FLOAT_ID', 'PROFILE_NUMBER', 'CYCLE_NUMBER', 'LATITUDE', 'LONGITUDE']
        available_cols = [col for col in sample_cols if col in df_profiles.columns]
        if available_cols:
            print(df_profiles[available_cols].head(3).to_string(index=False))
    
    # Save MEASUREMENTS.csv
    if all_measurements:
        df_measurements = pd.DataFrame(all_measurements)
        df_measurements.to_csv("MEASUREMENTS.csv", index=False)
        files_created.append(f"MEASUREMENTS.csv ({len(df_measurements)} measurements)")
        print(f"\n✅ MEASUREMENTS.csv saved: {len(df_measurements)} measurements")
        
        # Show sample
        print("   Sample data:")
        sample_cols = ['FLOAT_ID', 'PROFILE_NUMBER', 'LEVEL', 'PRES', 'TEMP', 'PSAL']
        available_cols = [col for col in sample_cols if col in df_measurements.columns]
        if available_cols:
            print(df_measurements[available_cols].head(5).to_string(index=False))
    
    return files_created

def main():
    """Main extraction function"""
    print("🌊 ARGO MULTI-FLOAT DATA EXTRACTOR v2.0")
    print("Based on data structure analysis")
    print("=" * 80)
    
    # Configuration
    float_ids = ["6903091", "6903006", "6903009", "6903010", "6903011", "6903014", "6903016", "6903018", "6903019", "6903020"]
    max_profiles = 20
    dac = "coriolis"
    
    print(f"🎯 Configuration:")
    print(f"   Float IDs: {float_ids}")
    print(f"   Max profiles per float: {max_profiles}")
    print(f"   DAC: {dac}")
    
    # Process all floats
    all_float_metadata, all_profile_data, all_measurements = process_multiple_floats(
        float_ids, dac=dac, max_profiles=max_profiles
    )
    
    # Save results
    files_created = save_to_csv_files(all_float_metadata, all_profile_data, all_measurements)
    
    # Final summary
    print(f"\n🎉 EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"📈 Final Results:")
    print(f"   Floats processed: {len(all_float_metadata)}")
    print(f"   Profiles extracted: {len(all_profile_data)}")
    print(f"   Measurements recorded: {len(all_measurements)}")
    
    if len(all_measurements) > 0 and len(all_profile_data) > 0:
        avg_measurements = len(all_measurements) / len(all_profile_data)
        print(f"   Average measurements per profile: {avg_measurements:.1f}")
    
    print(f"\n📁 Files created:")
    for file_info in files_created:
        print(f"   📄 {file_info}")
    
    print(f"\n💡 Next steps:")
    print(f"   - Check the CSV files for data quality")
    print(f"   - Import into your analysis software")
    print(f"   - Use FLOAT_ID as foreign key to link tables")

if __name__ == "__main__":
    main()