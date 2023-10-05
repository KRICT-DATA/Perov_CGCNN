from ase.io import read, write
from ase.io.cif import write_cif
pos = read('CONTCAR')
write_cif('CONTCAR.cif', pos)

