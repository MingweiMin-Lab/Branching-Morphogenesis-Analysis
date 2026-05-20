# Code for figures in "Multilevel branching lung organoids recapitulate epithelium lineage specification"  

The mouse airway data were downloaded from the National Toxicology Program of U.S Department of Health and Human Services (https://cebs-ext.niehs.nih.gov/cahs/report/lapd/web-download-links).  

The human airway data were downloaded from a Pulmonary-Tree-Repairing project (https://github.com/m3dv/pulmonary-tree-repairing?tab=readme-ov-file).   

The mouse salivary gland data were gathered from the paper "Inflationary theory of branching morphogenesis in the mouse salivary gland (2022)".  


Requirements:  
Python = 3.11  
Nexworkx = 3.4.2  
Matplotlib = 3.1.0  
Umap-learn = 0.5.7  

If you want to use the code for your own data, you need to save the data into hdf5 file and organize the tree structure as following:  

```
--info
	--edgeLinkList (the starting and ending point No.)
	--nEdge
	--nVertices
	--type (data type, i.e. Mouse Trachea)
	--unit (physical unit)
	--voxelSize
--edges
	--e0
		--length
		--pSource (the starting point No.)
		--pTarget (the ending point No.)
		--radius_avg (average radius)
		--x (x coordinates)
		--y (y coordinates)
		--z (z coordinates)
	--e1
	--eN
--vertices
	--v0
		--parent
		--nChild (total children number)
		--child1
		--vecIn (the edge direction pointing into this point)
		--vecOut1 (the edge direction pointing out to child1)
		--xyz
	--v1
		--parent
		--nChild (total children number)
		--child1
		--child2
		--vecIn (the edge direction pointing into this point)
		--vecOut1 (the edge direction pointing out to child1)
		--vecOut2 (the edge direction pointing out to child2)
		--xyz
	--vN
```