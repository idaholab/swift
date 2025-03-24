#!/usr/bin/env python3
#* This file is part of the MOOSE framework
#* https://www.mooseframework.org
#*
#* All rights reserved, see COPYRIGHT for full restrictions
#* https://github.com/idaholab/moose/blob/master/COPYRIGHT
#*
#* Licensed under LGPL 2.1, please see LICENSE for details
#* https://www.gnu.org/licenses/lgpl-2.1.html

"""
Create structured vtk grid from 2D or 3D numpy arrays
"""

import sys
import numpy as np
import vtk
from vtk.util import numpy_support

def create_vtk_from_arrays(structured_grid_matrices: list):
  """
  Create a VTK structured grid from a 2D or 3D numpy arrays

  Parameters:
  - structured_grid_matrices: List of numpy arrays

  Returns:
  - vtkMultiBlockDataSet object
  """
  # Determine dimensions
  # FOR NOW only 2D is supported
  # Create a structured grid
  shape = structured_grid_matrices[0].shape
  structured_grid = vtk.vtkStructuredGrid()
  structured_grid.SetDimensions(shape[0], shape[1], 1)

  # Create points
  points = vtk.vtkPoints()
  for i in range(shape[0]):
    for j in range(shape[1]):
      points.InsertNextPoint(i, j, 0)
  structured_grid.SetPoints(points)

  # Convert numpy arrays to vtk arrays
  binary_media_vtk = numpy_support.numpy_to_vtk(structured_grid_matrices[0].ravel(), deep=True, array_type=vtk.VTK_INT)
  pore_size_vtk = numpy_support.numpy_to_vtk(structured_grid_matrices[1].ravel(), deep=True, array_type=vtk.VTK_FLOAT)
  knudsen_number_vtk = numpy_support.numpy_to_vtk(structured_grid_matrices[2].ravel(), deep=True, array_type=vtk.VTK_FLOAT)

  # Set names for the arrays
  binary_media_vtk.SetName("BinaryMedia")
  pore_size_vtk.SetName("PoreSize")
  knudsen_number_vtk.SetName("KnudsenNumber")

  # Add arrays to the structured grid
  structured_grid.GetPointData().AddArray(binary_media_vtk)
  structured_grid.GetPointData().AddArray(pore_size_vtk)
  structured_grid.GetPointData().AddArray(knudsen_number_vtk)

  return structured_grid

if __name__ == "__main__":
  # Ensure enough arguments are provided
  if len(sys.argv) < 4:
    print("Usage: python3 create_vtk.py <file_path> <ny> <nx>")
    print("Optionally: python3 create_vtk.py <file_path> <ny> <nx> <output_directory>")
    sys.exit(1)

  file_path = sys.argv[1]
  ny = int(sys.argv[2])
  nx = int(sys.argv[3])
  """
  Each file must contain linearized 2D array in column first order
  """
  pore_image = np.loadtxt(file_path + '/pore.dat').reshape((ny, nx))
  kn = np.loadtxt(file_path + '/Kn.dat').reshape((ny, nx))
  local_pore_sizes = np.loadtxt(file_path + '/localporesize.dat').reshape((ny, nx))

  matrices = []
  matrices.append(pore_image)
  matrices.append(kn)
  matrices.append(local_pore_sizes)

  # Write
  file_path = "."
  if len(sys.argv) == 5:
      file_path = sys.argv[4]

  structured_grid = create_vtk_from_arrays(matrices)
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName(file_path)
  writer.SetInputData(structured_grid)
  writer.Write()
  print("Done")
