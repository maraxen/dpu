# ReFactored eVOLVER.py 

After attempting experimentation on the eVOLVER for many months, I felt the script could be further 
developed towards greater employability.

Several classes of objects were made with the goal of enhancing usability. 
Data processing was optimized to be more streamlined and up to date with the best available Python modules.

eVOLVER parameters are now managed via a Components class and standard methods which I anticipate will facilitate coding 
more multiplex experimentation.

New Features
--------
- **Options**:
  
    - custom_script object class containing experiment settings
    - it can require certain inputs every experiment
      while also taking keyword arguments for context-specific management

- **Vials**:
    - facilitates storing, recall, and comprehension of data 
    - as we increase complexity and the amounts of feedback we incorporate
    into our experimentation, this will help manage variation across 
    sets of sleeves
      
- **Components**:
     - object class relating to the different parameters of the eVOLVER 
     -  OD, temperature, pumps, spin, and light are handled through this

- **Calibration**: 
    - class containing the calibration file and methods to interpret it

Contribute
----------

- Issue Tracker: github.com/maraxen/dpu/issues
- Source Code: github.com/maraxen/dpu

Future Directions
--------
- validate Vial.restart() works properly
- eliminate redundancy
