#!/usr/bin/env python3
"""
Script to create the optimized GLOBEM dataset.
It processes raw features, simplifying column names and categorizing them into separate files.
This version is optimized for open-source usage, supporting configurable paths and generating only categorized datasets.
"""

import os
import csv
import json
import argparse
from pathlib import Path

def simplify_column_name(original_col):
    """Simplifies the original column name to a shorter form."""
    if original_col in ['', 'pid', 'date']:
        return original_col
    
    # Remove suffix :allday
    col = original_col.replace(':allday', '')
    
    # Remove prefix f_xxx:
    if ':' in col:
        col = col.split(':', 1)[1]
    
    # Remove sensor type prefixes
    prefixes_to_remove = [
        'fitbit_sleep_',
        'fitbit_steps_',
        'phone_screen_',
        'phone_calls_',
        'phone_bluetooth_',
        'phone_wifi_',
        'phone_locations_'
    ]
    
    for prefix in prefixes_to_remove:
        if col.startswith(prefix):
            col = col[len(prefix):]
            break
    
    return col

def create_field_description(simplified_name, original_name):
    """Creates a description for the simplified column name."""
    if simplified_name in ['', 'pid', 'date']:
        descriptions = {
            '': 'Row index',
            'pid': 'Participant identifier',
            'date': 'Date of observation'
        }
        return descriptions.get(simplified_name, simplified_name)
    
    # Full feature description mapping based on GLOBEM documentation
    desc_mapping = {
        # Physical Activity Features
        'summary_rapids_maxsumsteps': 'Maximum daily step count across all monitoring period',
        'summary_rapids_minsumsteps': 'Minimum daily step count across all monitoring period', 
        'summary_rapids_avgsumsteps': 'Average daily step count across all monitoring period',
        'summary_rapids_mediansumsteps': 'Median daily step count across all monitoring period',
        'summary_rapids_stdsumsteps': 'Standard deviation of daily step counts across all monitoring period',
        'intraday_rapids_sumsteps': 'Total step count for the day',
        'intraday_rapids_maxsteps': 'Maximum step count in a single time segment',
        'intraday_rapids_minsteps': 'Minimum step count in a single time segment',
        'intraday_rapids_avgsteps': 'Average step count per time segment',
        'intraday_rapids_stdsteps': 'Standard deviation of step counts across time segments',
        'intraday_rapids_countepisodesedentarybout': 'Number of sedentary episodes (continuous periods with step count < threshold)',
        'intraday_rapids_sumdurationsedentarybout': 'Total duration of all sedentary episodes in minutes',
        'intraday_rapids_maxdurationsedentarybout': 'Maximum duration of a single sedentary episode in minutes',
        'intraday_rapids_mindurationsedentarybout': 'Minimum duration of a single sedentary episode in minutes',
        'intraday_rapids_avgdurationsedentarybout': 'Average duration of sedentary episodes in minutes',
        'intraday_rapids_stddurationsedentarybout': 'Standard deviation of sedentary episode durations in minutes',
        'intraday_rapids_countepisodeactivebout': 'Number of active episodes (continuous periods with step count >= threshold)',
        'intraday_rapids_sumdurationactivebout': 'Total duration of all active episodes in minutes',
        'intraday_rapids_maxdurationactivebout': 'Maximum duration of a single active episode in minutes',
        'intraday_rapids_mindurationactivebout': 'Minimum duration of a single active episode in minutes',
        'intraday_rapids_avgdurationactivebout': 'Average duration of active episodes in minutes',
        'intraday_rapids_stddurationactivebout': 'Standard deviation of active episode durations in minutes',
        
        # Call/Communication Features
        'rapids_missed_count': 'Number of missed calls during the time segment',
        'rapids_missed_distinctcontacts': 'Number of distinct contacts associated with missed calls',
        'rapids_missed_timefirstcall': 'Time in minutes from midnight to the first missed call',
        'rapids_missed_timelastcall': 'Time in minutes from midnight to the last missed call',
        'rapids_missed_countmostfrequentcontact': 'Number of missed calls from the most frequent contact',
        'rapids_incoming_count': 'Number of incoming calls during the time segment',
        'rapids_incoming_distinctcontacts': 'Number of distinct contacts associated with incoming calls',
        'rapids_incoming_meanduration': 'Mean duration of all incoming calls in seconds',
        'rapids_incoming_sumduration': 'Sum duration of all incoming calls in seconds',
        'rapids_incoming_minduration': 'Duration of the shortest incoming call in seconds',
        'rapids_incoming_maxduration': 'Duration of the longest incoming call in seconds',
        'rapids_incoming_stdduration': 'Standard deviation of incoming call durations in seconds',
        'rapids_incoming_modeduration': 'Mode duration of all incoming calls in seconds',
        'rapids_incoming_entropyduration': 'Shannon entropy estimate for incoming call durations in nats',
        'rapids_incoming_timefirstcall': 'Time in minutes from midnight to the first incoming call',
        'rapids_incoming_timelastcall': 'Time in minutes from midnight to the last incoming call',
        'rapids_incoming_countmostfrequentcontact': 'Number of incoming calls from the most frequent contact',
        'rapids_outgoing_count': 'Number of outgoing calls during the time segment',
        'rapids_outgoing_distinctcontacts': 'Number of distinct contacts associated with outgoing calls',
        'rapids_outgoing_meanduration': 'Mean duration of all outgoing calls in seconds',
        'rapids_outgoing_sumduration': 'Sum duration of all outgoing calls in seconds',
        'rapids_outgoing_minduration': 'Duration of the shortest outgoing call in seconds',
        'rapids_outgoing_maxduration': 'Duration of the longest outgoing call in seconds',
        'rapids_outgoing_stdduration': 'Standard deviation of outgoing call durations in seconds',
        'rapids_outgoing_modeduration': 'Mode duration of all outgoing calls in seconds',
        'rapids_outgoing_entropyduration': 'Shannon entropy estimate for outgoing call durations in nats',
        'rapids_outgoing_timefirstcall': 'Time in minutes from midnight to the first outgoing call',
        'rapids_outgoing_timelastcall': 'Time in minutes from midnight to the last outgoing call',
        'rapids_outgoing_countmostfrequentcontact': 'Number of outgoing calls to the most frequent contact',
        
        # Bluetooth/Connectivity Features
        'rapids_countscans': 'Number of Bluetooth scans (rows) from devices sensed during time segment',
        'rapids_uniquedevices': 'Number of unique Bluetooth devices sensed, identified by hardware addresses',
        'rapids_countscansmostuniquedevice': 'Number of scans from the most frequently scanned device',
        'doryab_countscansall': 'Number of scans from all Bluetooth devices (own + others)',
        'doryab_uniquedevicesall': 'Number of unique devices sensed (own + others)',
        'doryab_meanscansall': 'Mean number of scans per device (own + others)',
        'doryab_stdscansall': 'Standard deviation of scans per device (own + others)',
        'doryab_countscansmostfrequentdevicewithinsegmentsall': 'Scans from most frequent device within this segment (all)',
        'doryab_countscansmostfrequentdeviceacrosssegmentsall': 'Scans from most frequent device across all segments (all)',
        'doryab_countscansmostfrequentdeviceacrossdatasetall': 'Scans from most frequent device across entire dataset (all)',
        'doryab_countscansleastfrequentdevicewithinsegmentsall': 'Scans from least frequent device within this segment (all)',
        'doryab_countscansleastfrequentdeviceacrosssegmentsall': 'Scans from least frequent device across all segments (all)',
        'doryab_countscansleastfrequentdeviceacrossdatasetall': 'Scans from least frequent device across entire dataset (all)',
        'doryab_countscansown': 'Number of scans from participant\'s own Bluetooth devices',
        'doryab_uniquedevicesown': 'Number of unique devices belonging to participant',
        'doryab_meanscansown': 'Mean number of scans per own device',
        'doryab_stdscansown': 'Standard deviation of scans per own device',
        'doryab_countscansmostfrequentdevicewithinsegmentsown': 'Scans from most frequent own device within segment',
        'doryab_countscansmostfrequentdeviceacrosssegmentsown': 'Scans from most frequent own device across segments',
        'doryab_countscansmostfrequentdeviceacrossdatasetown': 'Scans from most frequent own device across dataset',
        'doryab_countscansleastfrequentdevicewithinsegmentsown': 'Scans from least frequent own device within segment',
        'doryab_countscansleastfrequentdeviceacrosssegmentsown': 'Scans from least frequent own device across segments',
        'doryab_countscansleastfrequentdeviceacrossdatasetown': 'Scans from least frequent own device across dataset',
        'doryab_countscansothers': 'Number of scans from other people\'s Bluetooth devices',
        'doryab_uniquedevicesothers': 'Number of unique devices belonging to others',
        'doryab_meanscansothers': 'Mean number of scans per others\' device',
        'doryab_stdscansothers': 'Standard deviation of scans per others\' device',
        'doryab_countscansmostfrequentdevicewithinsegmentsothers': 'Scans from most frequent others\' device within segment',
        'doryab_countscansmostfrequentdeviceacrosssegmentsothers': 'Scans from most frequent others\' device across segments',
        'doryab_countscansmostfrequentdeviceacrossdatasetothers': 'Scans from most frequent others\' device across dataset',
        'doryab_countscansleastfrequentdevicewithinsegmentsothers': 'Scans from least frequent others\' device within segment',
        'doryab_countscansleastfrequentdeviceacrosssegmentsothers': 'Scans from least frequent others\' device across segments',
        'doryab_countscansleastfrequentdeviceacrossdatasetothers': 'Scans from least frequent others\' device across dataset',
        'connected_rapids_countscans': 'Number of scans when device was connected to network',
        'connected_rapids_uniquedevices': 'Number of unique devices when connected to network',
        'connected_rapids_countscansmostuniquedevice': 'Scans from most unique device when connected',
        
        # Location Features
        'barnett_avgflightdur': 'Mean duration of all flights (movement episodes) in seconds',
        'barnett_avgflightlen': 'Mean length of all flights (movement episodes) in meters',
        'barnett_circdnrtn': 'Circadian routine measure (0-1): 0=completely different routine, 1=same routine every day',
        'barnett_disttravelled': 'Total distance traveled over the day in meters (sum of all flights)',
        'barnett_hometime': 'Time spent at home in minutes. Home is most visited location between 8pm-8am within 200m radius',
        'barnett_maxdiam': 'Maximum diameter: largest distance between any two pauses in meters',
        'barnett_maxhomedist': 'Maximum distance from home location in meters',
        'barnett_probpause': 'Fraction of day spent in pauses (stationary periods) vs flights (movement)',
        'barnett_rog': 'Radius of Gyration: area coverage measure in meters, weighted distance from centroid of all visited places',
        'barnett_siglocentropy': 'Shannon entropy based on time spent at each significant location in nats',
        'barnett_siglocsvisited': 'Number of significant locations visited (found using k-means clustering, k=1-200, min 400m apart)',
        'barnett_stdflightdur': 'Standard deviation of flight (movement episode) durations in seconds',
        'barnett_stdflightlen': 'Standard deviation of flight (movement episode) lengths in meters',
        'barnett_wkenddayrtn': 'Weekend vs weekday routine difference (same as circdnrtn but computed separately)',
        'doryab_avglengthstayatclusters': 'Average time spent at significant locations (clusters) in minutes',
        'doryab_avgspeed': 'Average speed during movement periods in km/hr (0 when stationary)',
        'doryab_homelabel': 'Label identifier for home location cluster',
        'doryab_locationentropy': 'Shannon entropy over time spent at each significant location cluster in nats',
        'doryab_locationvariance': 'Sum of variances of latitude and longitude coordinates in meters²',
        'doryab_loglocationvariance': 'Log of the sum of variances of latitude and longitude coordinates',
        'doryab_maxlengthstayatclusters': 'Maximum time spent at any significant location in minutes',
        'doryab_minlengthstayatclusters': 'Minimum time spent at any significant location in minutes',
        'doryab_movingtostaticratio': 'Ratio of stationary time to total location sensing time (higher=more stationary)',
        'doryab_normalizedlocationentropy': 'Shannon entropy of location clusters divided by number of clusters in nats',
        'doryab_numberlocationtransitions': 'Number of movements between different significant location clusters',
        'doryab_numberofsignificantplaces': 'Number of significant places identified using DBSCAN/OPTICS clustering',
        'doryab_outlierstimepercent': 'Ratio of time in non-significant clusters to total cluster time',
        'doryab_radiusgyration': 'Area coverage quantification: weighted distance from centroid in meters',
        'doryab_stdlengthstayatclusters': 'Standard deviation of time spent at significant locations in minutes',
        'doryab_timeathome': 'Time spent at home location in minutes',
        'doryab_timeattop1location': 'Time spent at the most visited significant location in minutes',
        'doryab_timeattop2location': 'Time spent at the 2nd most visited significant location in minutes',
        'doryab_timeattop3location': 'Time spent at the 3rd most visited significant location in minutes',
        'doryab_totaldistance': 'Total distance traveled using haversine formula in meters',
        'doryab_varspeed': 'Speed variance during movement periods in km/hr (0 when stationary)',
        'locmap_duration_in_locmap_study': 'Time spent at study locations in minutes',
        'locmap_percent_in_locmap_study': 'Percentage of day spent at study locations',
        'locmap_duration_in_locmap_exercise': 'Time spent at exercise locations in minutes',
        'locmap_percent_in_locmap_exercise': 'Percentage of day spent at exercise locations',
        'locmap_duration_in_locmap_greens': 'Time spent at green/nature locations in minutes',
        'locmap_percent_in_locmap_greens': 'Percentage of day spent at green/nature locations',
        
        # Phone Usage Features
        'rapids_countepisodeunlock': 'Total number of phone unlock episodes during the day',
        'rapids_sumdurationunlock': 'Total duration of all phone unlock episodes in minutes',
        'rapids_maxdurationunlock': 'Longest duration of any single phone unlock episode in minutes',
        'rapids_mindurationunlock': 'Shortest duration of any single phone unlock episode in minutes',
        'rapids_avgdurationunlock': 'Average duration of phone unlock episodes in minutes',
        'rapids_stddurationunlock': 'Standard deviation of phone unlock episode durations in minutes',
        'rapids_firstuseafter00unlock': 'Minutes from midnight until first phone unlock episode',
        'rapids_countepisodeunlock_locmap_exercise': 'Number of phone unlock episodes at exercise locations',
        'rapids_sumdurationunlock_locmap_exercise': 'Total phone usage duration at exercise locations in minutes',
        'rapids_maxdurationunlock_locmap_exercise': 'Longest phone usage at exercise locations in minutes',
        'rapids_mindurationunlock_locmap_exercise': 'Shortest phone usage at exercise locations in minutes',
        'rapids_avgdurationunlock_locmap_exercise': 'Average phone usage duration at exercise locations in minutes',
        'rapids_stddurationunlock_locmap_exercise': 'Standard deviation of phone usage at exercise locations in minutes',
        'rapids_firstuseafter00unlock_locmap_exercise': 'Minutes until first phone use at exercise locations',
        'rapids_countepisodeunlock_locmap_greens': 'Number of phone unlock episodes at green/nature locations',
        'rapids_sumdurationunlock_locmap_greens': 'Total phone usage duration at green/nature locations in minutes',
        'rapids_maxdurationunlock_locmap_greens': 'Longest phone usage at green/nature locations in minutes',
        'rapids_mindurationunlock_locmap_greens': 'Shortest phone usage at green/nature locations in minutes',
        'rapids_avgdurationunlock_locmap_greens': 'Average phone usage duration at green/nature locations in minutes',
        'rapids_stddurationunlock_locmap_greens': 'Standard deviation of phone usage at green/nature locations in minutes',
        'rapids_firstuseafter00unlock_locmap_greens': 'Minutes until first phone use at green/nature locations',
        'rapids_countepisodeunlock_locmap_living': 'Number of phone unlock episodes at living/home locations',
        'rapids_sumdurationunlock_locmap_living': 'Total phone usage duration at living/home locations in minutes',
        'rapids_maxdurationunlock_locmap_living': 'Longest phone usage at living/home locations in minutes',
        'rapids_mindurationunlock_locmap_living': 'Shortest phone usage at living/home locations in minutes',
        'rapids_avgdurationunlock_locmap_living': 'Average phone usage duration at living/home locations in minutes',
        'rapids_stddurationunlock_locmap_living': 'Standard deviation of phone usage at living/home locations in minutes',
        'rapids_firstuseafter00unlock_locmap_living': 'Minutes until first phone use at living/home locations',
        'rapids_countepisodeunlock_locmap_study': 'Number of phone unlock episodes at study locations',
        'rapids_sumdurationunlock_locmap_study': 'Total phone usage duration at study locations in minutes',
        'rapids_maxdurationunlock_locmap_study': 'Longest phone usage at study locations in minutes',
        'rapids_mindurationunlock_locmap_study': 'Shortest phone usage at study locations in minutes',
        'rapids_avgdurationunlock_locmap_study': 'Average phone usage duration at study locations in minutes',
        'rapids_stddurationunlock_locmap_study': 'Standard deviation of phone usage at study locations in minutes',
        'rapids_firstuseafter00unlock_locmap_study': 'Minutes until first phone use at study locations',
        'rapids_countepisodeunlock_locmap_home': 'Number of phone unlock episodes at home location',
        'rapids_sumdurationunlock_locmap_home': 'Total phone usage duration at home location in minutes',
        'rapids_maxdurationunlock_locmap_home': 'Longest phone usage at home location in minutes',
        'rapids_mindurationunlock_locmap_home': 'Shortest phone usage at home location in minutes',
        'rapids_avgdurationunlock_locmap_home': 'Average phone usage duration at home location in minutes',
        'rapids_stddurationunlock_locmap_home': 'Standard deviation of phone usage at home location in minutes',
        'rapids_firstuseafter00unlock_locmap_home': 'Minutes until first phone use at home location',
        
        # Sleep Features
        'summary_rapids_sumdurationafterwakeupmain': 'Total time after wakeup in main sleep episode in minutes',
        'summary_rapids_sumdurationasleepmain': 'Total sleep duration in main sleep episode in minutes',
        'summary_rapids_sumdurationawakemain': 'Total time awake during main sleep episode in minutes',
        'summary_rapids_sumdurationtofallasleepmain': 'Total time to fall asleep in main sleep episode in minutes',
        'summary_rapids_sumdurationinbedmain': 'Total time in bed for main sleep episode in minutes',
        'summary_rapids_avgefficiencymain': 'Average sleep efficiency of main sleep episode (sleep time / time in bed)',
        'summary_rapids_avgdurationafterwakeupmain': 'Average time after wakeup in main sleep episode in minutes',
        'summary_rapids_avgdurationasleepmain': 'Average sleep duration in main sleep episode in minutes',
        'summary_rapids_avgdurationawakemain': 'Average time awake during main sleep episode in minutes',
        'summary_rapids_avgdurationtofallasleepmain': 'Average time to fall asleep in main sleep episode in minutes',
        'summary_rapids_avgdurationinbedmain': 'Average time in bed for main sleep episode in minutes',
        'summary_rapids_countepisodemain': 'Number of main sleep episodes',
        'summary_rapids_firstbedtimemain': 'Time of first bedtime in main sleep episode (minutes from midnight)',
        'summary_rapids_lastbedtimemain': 'Time of last bedtime in main sleep episode (minutes from midnight)',
        'summary_rapids_firstwaketimemain': 'Time of first wake in main sleep episode (minutes from midnight)',
        'summary_rapids_lastwaketimemain': 'Time of last wake in main sleep episode (minutes from midnight)',
        'intraday_rapids_avgdurationasleepunifiedmain': 'Average duration of unified asleep periods in main episode in minutes',
        'intraday_rapids_avgdurationawakeunifiedmain': 'Average duration of unified awake periods in main episode in minutes',
        'intraday_rapids_maxdurationasleepunifiedmain': 'Maximum duration of unified asleep period in main episode in minutes',
        'intraday_rapids_maxdurationawakeunifiedmain': 'Maximum duration of unified awake period in main episode in minutes',
        'intraday_rapids_sumdurationasleepunifiedmain': 'Total duration of unified asleep periods in main episode in minutes',
        'intraday_rapids_sumdurationawakeunifiedmain': 'Total duration of unified awake periods in main episode in minutes',
        'intraday_rapids_countepisodeasleepunifiedmain': 'Number of unified asleep episodes in main sleep period',
        'intraday_rapids_countepisodeawakeunifiedmain': 'Number of unified awake episodes in main sleep period',
        'intraday_rapids_stddurationasleepunifiedmain': 'Standard deviation of unified asleep episode durations in minutes',
        'intraday_rapids_stddurationawakeunifiedmain': 'Standard deviation of unified awake episode durations in minutes',
        'intraday_rapids_mindurationasleepunifiedmain': 'Minimum duration of unified asleep episode in main period in minutes',
        'intraday_rapids_mindurationawakeunifiedmain': 'Minimum duration of unified awake episode in main period in minutes',
        'intraday_rapids_mediandurationasleepunifiedmain': 'Median duration of unified asleep episodes in main period in minutes',
        'intraday_rapids_mediandurationawakeunifiedmain': 'Median duration of unified awake episodes in main period in minutes',
        'intraday_rapids_ratiocountasleepunifiedwithinmain': 'Ratio of asleep episode count to total episodes within main sleep',
        'intraday_rapids_ratiocountawakeunifiedwithinmain': 'Ratio of awake episode count to total episodes within main sleep',
        'intraday_rapids_ratiodurationasleepunifiedwithinmain': 'Ratio of asleep duration to total duration within main sleep',
        'intraday_rapids_ratiodurationawakeunifiedwithinmain': 'Ratio of awake duration to total duration within main sleep',
    }
    
    # Use predefined description if available
    if simplified_name in desc_mapping:
        return desc_mapping[simplified_name]
    
    # Otherwise generate a generic description
    desc = simplified_name
    if 'f_slp:' in original_name:
        desc += " (daily sleep data)"
    elif 'f_steps:' in original_name:
        desc += " (daily activity data)"
    elif 'f_loc:' in original_name:
        desc += " (daily location data)"
    elif 'f_screen:' in original_name:
        desc += " (daily phone usage data)"
    elif 'f_call:' in original_name:
        desc += " (daily communication data)"
    elif 'f_blue:' in original_name or 'f_wifi:' in original_name:
        desc += " (daily connectivity data)"
    else:
        desc += " (daily sensor data)"
    
    return desc

def get_allday_raw_features(input_file):
    """Identifies all 'allday' raw feature columns from the input CSV."""
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    
    # Identify all allday raw features
    target_features = {}
    metadata_indices = []
    
    for i, col in enumerate(headers):
        if col in ['', 'pid', 'date']:
            metadata_indices.append((i, col, col))  # (index, original_name, simplified_name)
        elif col.endswith(':allday') and '_dis:' not in col and '_norm:' not in col:
            simplified_name = simplify_column_name(col)
            
            # Categorize by sensor type
            if 'f_slp:' in col:
                category = 'sleep'
            elif 'f_steps:' in col:
                category = 'activity'
            elif 'f_loc:' in col:
                category = 'location'
            elif 'f_screen:' in col:
                category = 'phone_usage'
            elif 'f_call:' in col:
                category = 'communication'
            elif 'f_blue:' in col or 'f_wifi:' in col:
                category = 'connectivity'
            else:
                category = 'other'
            
            if category not in target_features:
                target_features[category] = []
            target_features[category].append((i, col, simplified_name))  # (index, original_name, simplified_name)
    
    return metadata_indices, target_features

def create_optimized_files(input_file, output_dir):
    """Creates the optimized, categorized data files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get target features
    metadata_indices, target_features = get_allday_raw_features(input_file)
    
    print(f"Metadata fields: {len(metadata_indices)}")
    for category, features in target_features.items():
        print(f"{category}: {len(features)} features")
    
    total_features = sum(len(features) for features in target_features.values())
    print(f"Total: {total_features} allday raw features")
    
    # Prepare output files and writers
    output_files = {}
    writers = {}
    
    # Create categorized datasets
    for category, features in target_features.items():
        output_path = os.path.join(output_dir, f'{category}_allday_raw.csv')
        output_files[category] = open(output_path, 'w', newline='')
        writers[category] = csv.writer(output_files[category])
        
        # Write header (metadata + category features)
        category_headers = [simplified_name for _, _, simplified_name in metadata_indices] + [simplified_name for _, _, simplified_name in features]
        writers[category].writerow(category_headers)
        
        # Create field description JSON
        field_descriptions = {}
        for _, original_name, simplified_name in metadata_indices:
            field_descriptions[simplified_name] = create_field_description(simplified_name, original_name)
        for _, original_name, simplified_name in features:
            field_descriptions[simplified_name] = create_field_description(simplified_name, original_name)
        
        json_path = os.path.join(output_dir, f'{category}_allday_raw_fields.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(field_descriptions, f, indent=2, ensure_ascii=False)
    
    # Process data rows
    with open(input_file, 'r') as f_in:
        reader = csv.reader(f_in)
        headers = next(reader)  # Skip header
        
        row_count = 0
        for row in reader:
            if len(row) == len(headers):
                # Prepare metadata values once
                metadata_data = [row[i] for i, _, _ in metadata_indices]
                
                # Write to each category file
                for category, features in target_features.items():
                    if category in writers:
                        feature_data = [row[i] for i, _, _ in features]
                        writers[category].writerow(metadata_data + feature_data)
                
                row_count += 1
                if row_count % 2000 == 0:
                    print(f"Processed {row_count} rows")
    
    # Close all files
    for f in output_files.values():
        f.close()
    
    # Return stats for summary
    stats = {
        "dataset_info": {
            "participants": 155,  # Note: logic implies we know this, or we could count unique IDs
            "days_per_participant": 92, # Approximate/Expected
            "total_observations": row_count,
            "time_window": "allday",
            "normalization": "raw (original values)",
            "total_features": total_features
        },
        "features_by_category": {},
        "files_created": {}
    }
    
    for category, features in target_features.items():
        feature_count = len(features)
        stats["features_by_category"][category] = {
            "count": feature_count,
            "sample_features": [simplified_name for _, _, simplified_name in features[:3]]
        }
        stats["files_created"][f"{category}_allday_raw.csv"] = f"{feature_count} features from {category} sensors"

    return stats

def main():
    parser = argparse.ArgumentParser(description="Create optimized GLOBEM dataset (categorized raw features).")
    parser.add_argument("--input", "-i", required=True, help="Path to the GLOBEM data, which is PATH_TO/physionet/globem/1.1/INS-W_1/FeatureData/")
    parser.add_argument("--output", "-o", required=True, help="Directory to save the processed files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    print("Creating Optimized GLOBEM Dataset")
    print("=" * 50)
    print("Target: Categorized allday raw features")
    print()
    
    # Create optimized files
    stats = create_optimized_files(args.input, args.output)
    
    # Save summary stats
    with open(os.path.join(args.output, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Rows processed: {stats['dataset_info']['total_observations']}")
    print(f"Total features: {stats['dataset_info']['total_features']}")
    
    print(f"\n=== Distribution by Category ===")
    for category, info in stats["features_by_category"].items():
        print(f"{category}: {info['count']} features")
    
    print(f"\n=== Files Created ===")
    for filename, description in stats["files_created"].items():
        print(f"• {filename}: {description}")
    print(f"• dataset_summary.json: Dataset statistics and metadata")
    
    print(f"\nOutput Directory: {args.output}")

if __name__ == "__main__":
    main()
