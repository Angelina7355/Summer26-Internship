##################################################
#        Environment Setup & Verification        #
##################################################
import sys
arcpy.AddMessage("Current Python executable:")
arcpy.AddMessage(sys.executable)


##################################################
#              Global Definitions                #
##################################################


# ----------------------------------------
# Classes
# ----------------------------------------

# Class dictionary
CLASSES = {
    "WATER": 0,
    "WETLANDS": 1,
    "TREE CANOPY": 2,
    "SHRUBLAND": 3,
    "LOW VEGETATION": 4,
    "BARREN": 5,
    "STRUCTURES": 6,
    "IMPERVIOUS SURFACES": 7,
    "IMPERVIOUS ROADS": 8,
    "NO DATA": 255
}

# Aliases for readability
WATER = CLASSES["WATER"]
WETLANDS = CLASSES["WETLANDS"]
TREE_CANOPY = CLASSES["TREE CANOPY"]
SHRUBLAND = CLASSES["SHRUBLAND"]
LOW_VEGETATION = CLASSES["LOW VEGETATION"]
BARREN = CLASSES["BARREN"]
STRUCTURES = CLASSES["STRUCTURES"]
IMPERVIOUS_SURFACES = CLASSES["IMPERVIOUS SURFACES"]
IMPERVIOUS_ROADS = CLASSES["IMPERVIOUS ROADS"]
NO_DATA = CLASSES["NO DATA"]


# ----------------------------------------
# Internal Datasets
# ----------------------------------------
USA_RASTER_NAME = "NAIP_USDA_CONUS_PRIME"


##################################################
#                Helper Functions                #
##################################################


# ----------------------------------------
# NDVI & DL Functions
# ----------------------------------------

def compute_ndvi_and_classify(input_raster):
    raster = arcpy.Raster(input_raster)
    
    # Convert to NumPy for pixel-wise NDVI computation
    arr = arcpy.RasterToNumPyArray(raster)
    
    # Define variables for each of the 4 bands (Red, Green, Blue, & Infrared) and the no-data mask
    r = arr[0].astype(float)
    g = arr[1].astype(float)
    b = arr[2].astype(float)
    nir = arr[3].astype(float)
    
    # Build NoData mask
    nodata_mask = (r == raster.noDataValue)
        
    # Compute NDVI for each pixel
    ndvi = (nir - r) / (nir + r + 1e-5)
    ndvi = np.nan_to_num(ndvi, nan=0)
        
    # Classify each pixel
    ndvi_class = np.zeros(ndvi.shape, dtype=np.uint8)
    ndvi_class[ndvi < 0.2] = IMPERVIOUS_ROADS                    # Roads / built
    ndvi_class[(ndvi >= 0.2) & (ndvi < 0.5)] = LOW_VEGETATION    # Grass / low vegetation
    ndvi_class[ndvi >= 0.5] = TREE_CANOPY                        # Trees / dense vegetation
                       
    # Apply NoData Mask
    ndvi_class[nodata_mask] = NO_DATA
    
    arcpy.AddMessage(f"NDVI classes: {np.unique(ndvi_class)}")

    return ndvi, ndvi_class


def run_dl_model(input_raster, model_path):
    raster = arcpy.Raster(input_raster)
    
    # Extract RGB only (bands 1–3) for input into DL model
    rgb_raster = arcpy.management.CompositeBands(
        [
            f"{raster}/Band_1",
            f"{raster}/Band_2",
            f"{raster}/Band_3"
        ],
        "in_memory/rgb_composite"
    )
    
    rgb_raster = arcpy.Raster(rgb_raster)

    result = ClassifyPixelsUsingDeepLearning(rgb_raster, model_path)
    return result


def fuse_results(ndvi_class, dl_class):
    final = dl_class.copy()
    
    # NDVI mask encompassing all vegetation pixels
    veg_mask = (
        (ndvi_class == LOW_VEGETATION) |
        (ndvi_class == TREE_CANOPY)
    )

    # DL mask encompassing all non-vegetation pixels
    NON_VEG_CLASSES = [IMPERVIOUS_ROADS, IMPERVIOUS_SURFACES, STRUCTURES, BARREN]
    nonveg_dl_mask = np.isin(dl_class, NON_VEG_CLASSES)
    
    # Only fix pixels where DL got vegetation wrong
    correction_mask = veg_mask & nonveg_dl_mask

    # Assign NDVI vegetation class
    final[correction_mask] = ndvi_class[correction_mask]
    
    # Preserve no-data pixels
    nodata_mask = (ndvi_class == NO_DATA)
    final[nodata_mask] = NO_DATA
    
    arcpy.AddMessage(f"Final classes: {np.unique(final)}")

    return final


# ----------------------------------------
# Raster Input Functions
# ----------------------------------------

def get_polygon(pipeline_line, easement_points, output_fc):
    # Get polygon based on pipeline and closest easement point
    arcpy.AddMessage("Buffering pipeline selection...")
    
    # Find closest easement point to specified pipeline
    arcpy.analysis.Near(
        in_features=pipeline_line, 
        near_features=easement_points
    )
    
    # Define default WIDTH
    width = 20  # default = 20 yards (60 feet)
    
    # Get object ID (NEAR_FID) of closest easement point
    near_fid = None
    with arcpy.da.SearchCursor(pipeline_line, ["NEAR_FID"]) as cursor:
        for row in cursor:
            near_fid = row[0]

    # Get WIDTH from closest easement point
    where_clause = f"OBJECTID = {near_fid}"
    
    with arcpy.da.SearchCursor(
        easement_points,
        ["WIDTH"],
        where_clause
    ) as cursor:
        for row in cursor:
            width = row[0]
    
    arcpy.AddMessage(f"Using easement width: {width} yards")
    
    # Buffer pipeline line to get polygon output
    polygon = arcpy.analysis.Buffer(
        in_features=pipeline_line,
        out_feature_class=output_fc,
        buffer_distance_or_field=f"{width} Yards",
        line_side="FULL",
        line_end_type="FLAT",
        dissolve_option="ALL"
    )

    return polygon


def clip_raster_to_polygon(active_map, raster_path, input_polygon):
    # Specify active map for sourcing raster layer
    m = active_map

    # Get raster path
    input_raster = None
    for lyr in m.listLayers():
        if lyr.name == raster_path:
            input_raster = arcpy.Raster(lyr.dataSource)

    if input_raster is None:
        arcpy.AddError(f"Layer '{raster_path}' not found")
        raise ValueError(f"Layer '{raster_path}' not found")
        
    # Clip the inputted polygon to the NAIP raster imagery
    arcpy.AddMessage("Clipping raster to polygon extent...")
    clipped_raster = arcpy.management.Clip(
        input_raster,
        "#",
        "in_memory/temp_clip",
        input_polygon,
        "#",
        "ClippingGeometry"
    )

    return clipped_raster


# ----------------------------------------
# Raster Output Functions
# ----------------------------------------

def save_raster(input_raster_path, raster_classifications, symbology_dir, output_path):
    # Preserve and save output as spatial reference (based on the input's spatial data)
    arcpy.AddMessage("Saving output raster...")
    os.makedirs(output_path, exist_ok=True)
    
    raster_obj = arcpy.Raster(input_raster_path)

    out_raster = arcpy.NumPyArrayToRaster(
        raster_classifications,
        lower_left_corner=raster_obj.extent.lowerLeft,
        x_cell_size=raster_obj.meanCellWidth,
        y_cell_size=raster_obj.meanCellHeight,
        value_to_nodata=NO_DATA
    )
    
    # Assign projection (coordinates)
    arcpy.DefineProjection_management(out_raster, raster_obj.spatialReference)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure no issues with data type or NO_DATA
    out_raster = arcpy.sa.Int(out_raster)
    out_raster = arcpy.sa.SetNull(out_raster == NO_DATA, out_raster)

    # Save output raster to disk
    temp_raster = "in_memory/temp_raster"

    arcpy.AddMessage("Writing intermediate raster...")
    arcpy.management.CopyRaster(out_raster, temp_raster)

    arcpy.AddMessage("Writing final output raster...")
    
    arcpy.management.CopyRaster(
        temp_raster,
        output_path,
        pixel_type="8_BIT_UNSIGNED"
    )

    arcpy.AddMessage("Raster save complete")
    
    return output_path


##################################################
#           Easement Analysis Pipeline           #
##################################################

def analyze_easement(clipped_raster, model_path):
    # Run NDVI pipeline (NumPy)
    arcpy.AddMessage("Running NDVI pipeline...")
    ndvi, ndvi_class = compute_ndvi_and_classify(clipped_raster)

    # Run deep learning model (ArcPy)
    # arcpy.AddMessage("Running deep learning model...")
    # dl_raster = run_dl_model(clipped_raster, model_path)

    # # Convert raster to numPy array
    # arcpy.AddMessage("Converting DL output to NumPy...")
    # dl_array = arcpy.RasterToNumPyArray(
    #     arcpy.Raster(dl_raster), 
    #     nodata_to_value=NO_DATA
    # )
    dl_array = ndvi_class.copy()

    # Fuse results
    arcpy.AddMessage("Fusing NDVI and DL results...")
    final_classifications = fuse_results(ndvi_class, dl_array)
    
    arcpy.AddMessage("Easement analysis complete")
    return final_classifications


##################################################
#                 Main Execution                 #
##################################################


# ----------------------------------------
# Imports & Setup
# ----------------------------------------

# Imports
arcpy.AddMessage("Starting imports...")
import os
import arcpy
import numpy as np
from arcgis.gis import GIS
from arcpy.sa import ExtractByMask
from arcpy.ia import ClassifyPixelsUsingDeepLearning

# Global setting
arcpy.env.overwriteOutput = True
arcpy.AddMessage("Imports complete")


# ----------------------------------------
# Main Function
# ----------------------------------------

def main(pipeline_selection, output_raster):
    arcpy.AddMessage("Starting connect to GIS...")
    gis = GIS("home")
    arcpy.AddMessage("GIS connection complete")
    
    # Directory structure
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    project_dir = os.path.dirname(aprx.filePath)

    data_dir = os.path.join(project_dir, "data")
    dl_model_dir = os.path.join(project_dir, "dl_model")
    symbology_dir = os.path.join(project_dir, "symbology")
    
    # Map reference
    m = aprx.activeMap

    # Input validation
    input_line = pipeline_selection

    count = int(arcpy.management.GetCount(input_line)[0])
    if count == 0:
        arcpy.AddMessage("No pipeline features selected")
        raise ValueError("No pipeline features selected")

    # Internal data sources
    easement_points = os.path.join(
        project_dir,
        "ROW Project.gdb",
        "easement_points"
    )

    buffer_fc = os.path.join("in_memory", "easement_buffer")    # temporary data


    # ----------------------------------------
    # Processing
    # ----------------------------------------

    arcpy.AddMessage("Processing started...")

    # Get polygon based on pipeline input
    polygon = get_polygon(input_line, easement_points, buffer_fc)
    
    # Clip raster to pipeline polygon
    clipped_raster = clip_raster_to_polygon(m, USA_RASTER_NAME, polygon)

    # Get deep learning model package path
    model_path = os.path.join(dl_model_dir, "HighResolutionLandCoverClassification_USA.dlpk")
    
    # Run easement analysis pipeline
    raster_classifications = analyze_easement(clipped_raster, model_path)
    
    # Save classified raster to notebook directory
    output_raster_path = save_raster(
        USA_RASTER_NAME, 
        raster_classifications, 
        symbology_dir, 
        output_raster
    )

    arcpy.AddMessage("Processing complete")
    
    return output_raster_path


# ----------------------------------------
# Run Main
# ----------------------------------------
if __name__ == "__main__":
    pipeline_selection = arcpy.GetParameter(0)
    output_raster = arcpy.GetParameterAsText(1)
    main(pipeline_selection, output_raster)


