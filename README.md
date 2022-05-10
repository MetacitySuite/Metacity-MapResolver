# Metacity-MapResolver
Pipeline for reading FCD dynamic data and transforming it to .sim format.

Structure:
.
├── code
│   ├── ceda_file.py
│   ├── ceda_mapper.py
│   ├── exporter.py
│   ├── fcd_bd_file.py
│   ├── fcd_file.py
│   ├── linker.py
│   ├── ndic_export.py
│   ├── tmc_file.py
│   └── transform.py
├── data
│   ├── CEDA
│   ├── csv
│   ├── FCD
│   ├── out
│   ├── png
│   └── shp
└── README.md


Pipeline:

1. fcd_bd_file.py -- reads FCD data, removes redundant data and stores it to smaller chunks

2. ceda_mapper.py -- reads map data, creates graph for mapping, iteratively maps FCD data onto realistic geometry

3. optional - exporter.py -- outputs fcd data as a csv file of geometries with a list of attributes: times and speeds

4. transform.py -- transforms (generates agents for each segment) mapped fcd data to .sim file 
