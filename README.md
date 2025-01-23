# SandAlas_extraction_3D
Extract 3D microstructure from the [Sand Atlas database](https://sand-atlas.scigem.com/).

## How to use
The file <i>main_extraction_3D.py</i> is called with the different parameters. <br>
Then, it called the [YADE](https://yade-dem.org/doc/) file <i>dem_base_3D.py</i> to generate a vtk file (for visualization).

A 2D version of this algorithm is available [here](https://github.com/AlexSacMorane/SandAtlas_extraction_2D).

## Relevant parameters
- i_x_min/i_y_min/i_z_min/i_x_max/i_y_max/i_z_max define the extraction window
- m_size defines the pixel size (must be similar than the database)


