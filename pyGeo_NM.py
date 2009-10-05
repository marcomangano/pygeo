'''
pyGeo

pyGeo is a (fairly) complete geometry surfacing engine. It performs
multiple functions including producing surfaces from cross sections,
fitting groups of surfaces with continutity constraints and has
built-in design variable handling. The actual b-spline surfaces are of
the pySpline surf_spline type. See the individual functions for
additional information

Copyright (c) 2009 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 26/05/2009$
s

Developers:
-----------
- Gaetan Kenway (GKK)
- Graeme Kennedy (GJK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2009)
'''

__version__ = '$Revision: $'

# =============================================================================
# Standard Python modules
# =============================================================================

import os, sys, string, copy, pdb, time

# =============================================================================
# External Python modules
# =============================================================================

from numpy import sin, cos, linspace, pi, zeros, where, hstack, mat, array, \
    transpose, vstack, max, dot, sqrt, append, mod, ones, interp, meshgrid, \
    real, imag, dstack, floor, size, reshape, arange,alltrue,cross

from numpy.linalg import lstsq,inv
#from scipy import io #Only used for debugging

try:
    import petsc4py
    # petsc4py.init(sys.argv)
    from petsc4py import PETSc
    
    USE_PETSC = True
    #USE_PETSC = False
    print 'PETSc4py is available. Least Square Solutions will be performed \
with PETSC'
    version = petsc4py.__version__
    vals = string.split(version,'.')
    PETSC_MAJOR_VERSION = int(vals[0])
    PETSC_MINOR_VERSION = int(vals[1])
    PETSC_UPDATE        = int(vals[2])
except:
    print 'PETSc4py is not available. Least Square Solutions will be performed\
with LAPACK (Numpy Least Squares)'
    USE_PETSC = False

# =============================================================================
# Extension modules
# =============================================================================

# pySpline Utilities
import pySpline

try:
    import csm_pre
    USE_CSM_PRE = True
    print 'CSM_PRE is available. Surface associations can be performed'
except:
    print 'CSM_PRE is not available. Surface associations cannot be performed'
    USE_CSM_PRE = False

from geo_utils import *


# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,init_type,*args, **kwargs):
        
        '''Create an instance of the geometry object. The initialization type,
        init_type, specifies what type of initialization will be
        used. There are currently 4 initialization types: plot3d,
        iges, lifting_surface and acdt_geo

        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d',file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined surfaces
 

        'iges',file_name = 'file_name.igs': Load the surface patches
        from an iges file to create splined surfaes.

        
        'lifting_surface',xsections=airfoil_list,scale=chord,offset=offset 
         Xsec=X,rot=rot

         Mandatory Arguments:
              
              xsections: List of the cross section coordinate files
              scsale   : List of the scaling factor for cross sections
              offset   : List of x-y offset to apply BEFORE scaling
              Xsec     : List of spatial coordinates as to the placement of 
                         cross sections
              rot      : List of x-y-z rotations to apply to cross sections

        Optional Arguments:

              breaks   : List of ZERO-BASED index locations where to break 
                         the wing into separate surfaces
              nsections: List of length breaks+1 which specifies the number
                         of control points in that section
              section_spacing : List of lenght breaks + 1 containing lists of
                         length nections which specifiy the spanwise spacing 
                         of control points
              fit_type : strig of either 'lms' or 'interpolate'. Used to 
                         initialize the surface patches
              Nctlu    : Number of control points in the chord-wise direction
              Nfoil    : Common number of data points extracted from cross
                         section file. Points are linearly interpolated to 
                         match this value
        
        'acdt_geo',acdt_geo=object : Load in a pyGeometry object and
        use the aircraft components to create surfaces.
        '''
        
        # First thing to do is to check if we want totally silent
        # operation i.e. no print statments

        if 'no_print' in kwargs:
            self.NO_PRINT = kwargs['no_print']
        else:
            self.NO_PRINT = False
        # end if
        
        if not self.NO_PRINT:
            print ' '
            print '------------------------------------------------'
            print 'pyGeo Initialization Type is: %s'%(init_type)
            print '------------------------------------------------'

        #------------------- pyGeo Class Atributes -----------------

        self.ref_axis       = [] # Reference Axis list
        self.ref_axis_con   = [] # Reference Axis connection list
        self.DV_listGlobal  = [] # Global Design Variable List
        self.DV_listNormal  = [] # Normal Design Variable List
        self.DV_listLocal   = [] # Local Design Variable List
        self.DV_namesGlobal = {} # Names of Global Design Variables
        self.DV_namesNormal = {} # Names of Normal Design Variables
        self.DV_namesLocal  = {} # Names of Local Design Variables
        self.petsc_coef = None # Global vector of PETSc coefficients
        self.J  = None           # Jacobian for full surface fitting
        self.dCoefdx  = None     # Derivative of control points wrt
                                 # design variables
        self.dPtdCoef = None     # Derivate of surface points wrt
                                 # control points
        self.dPtdx    = None     # Multiplication of above matricies,
                                 # derivative of surface points wrt
                                 # design variables
        self.con = None          # List of edge connection objects
        self.node_con = None     # The node connectivity list
        self.g_index = None      # Global Index: This is the length of
                                 # the reduced set of control points
                                 # and contains a list that points to
                                 # the surface and index of each point
                                 # that is logically the same
        self.l_index = None      # A entry for each surface (Nu,Nv)
                                 # which points to the the index in
                                 # the global list that is its master
                                 # (driving coefficient)

        self.surfs = []          # The list of surface (pySpline surf)
                                 # objects
        self.nSurf = None        # The total number of surfaces
        self.coef  = None        # The global (reduced) set of control
                                 # points
        self.l_surfs = []        # Logical Surfaces: List of list of
                                 # surfaces that can be thought of as
                                 # connected.

        # --------------------------------------------------------------

        if init_type == 'plot3d':
            assert 'file_name' in kwargs,'file_name must be specified as \
file_name=\'filename\' for plot3d init_type'
            self._readPlot3D(kwargs['file_name'],args,kwargs)

        elif init_type == 'iges':
            assert 'file_name' in kwargs,'file_name must be specified as \
file_name=\'filename\' for iges init_type'
            self._readIges(kwargs['file_name'],args,kwargs)

        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args,**kwargs)

        elif init_type == 'acdt_geo':
            self._init_acdt_geo(*args,**kwargs)
        elif init_type == 'create':
            # Don't do anything 
            pass
        else:
            print 'Unknown init type. Valid Init types are \'plot3d\', \
\'iges\',\'lifting_surface\' and \'acdt\''
            sys.exit(0)

        return

# ----------------------------------------------------------------------------
#               Initialization Type Functions
# ----------------------------------------------------------------------------

    def _readPlot3D(self,file_name,*args,**kwargs):

        '''Load a plot3D file and create the splines to go with each patch'''
        
        if not self.NO_PRINT:
            print 'Loading plot3D file: %s ...'%(file_name)

        f = open(file_name,'r')
        nSurf = int(f.readline())         # First load the number of patches

        if not self.NO_PRINT:
            print 'nSurf = %d'%(nSurf)

        patchSizes = readNValues(f,nSurf*3,'int')
        patchSizes = patchSizes.reshape([nSurf,3])

        assert patchSizes[:,2].all() == 1, \
            'Error: Plot 3d does not contain only surface patches.\
 The third index (k) MUST be 1.'

        # Total points
        nPts = 0
        for i in xrange(nSurf):
            nPts += patchSizes[i,0]*patchSizes[i,1]

        if not self.NO_PRINT:
            print 'Number of Surface Points = %d'%(nPts)

        dataTemp = readNValues(f,3*nPts,'float')
        
        f.close() # Done with the file

        # Post Processing
        patches = []
        counter = 0

        for isurf in xrange(nSurf):
            patches.append(zeros([patchSizes[isurf,0],patchSizes[isurf,1],3]))
            for idim in xrange(3):
                for j in xrange(patchSizes[isurf,1]):
                    for i in xrange(patchSizes[isurf,0]):
                        patches[isurf][i,j,idim] = dataTemp[counter]
                        counter += 1
                    # end for
                # end for
            # end for
        # end for

        # Now create a list of spline objects:
        surfs = []
        for isurf in xrange(nSurf):
            surfs.append(pySpline.surf_spline(task='lms',X=patches[isurf],\
                                                  ku=4,kv=4,Nctlu=8,Nctlv=8,\
                                                  no_print=self.NO_PRINT))

        self.surfs = surfs
        self.nSurf = nSurf
        return

    def _readIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        if not self.NO_PRINT:
            print 'File Name is: %s'%(file_name)
        f = open(file_name,'r')
        file = []
        for line in f:
            line = line.replace(';',',')  #This is a bit of a hack...
            file.append(line)
        f.close()
        
        start_lines   = int((file[-1][1:8]))
        general_lines = int((file[-1][9:16]))
        directory_lines = int((file[-1][17:24]))
        parameter_lines = int((file[-1][25:32]))

        #print start_lines,general_lines,directory_lines,parameter_lines
        
        # Now we know how many lines we have to deal 

        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        # Directory lines is ALWAYS a multiple of 2
        for i in xrange(directory_lines/2): 
            if int(file[2*i + dir_offset][0:8]) == 128:
                start = int(file[2*i + dir_offset][8:16])
                num_lines = int(file[2*i + 1 + dir_offset][24:32])
                surf_list.append([start,num_lines])
            # end if
        # end for
        self.nSurf = len(surf_list)

        if not self.NO_PRINT:
            print 'Found %d surfaces in Iges File.'%(self.nSurf)

        self.surfs = [];
        #print surf_list
        weight = []
        for isurf in xrange(self.nSurf):  # Loop over our patches
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[isurf][0]+dir_offset+directory_lines-1 

            for i in xrange(surf_list[isurf][1]):
                aux = string.split(file[i+para_offset][0:69],',')
                for j in xrange(len(aux)-1):
                    data.append(float(aux[j]))
                # end for
            # end for
            
            # Now we extract what we need
            Nctlu = int(data[1]+1)
            Nctlv = int(data[2]+1)
            ku    = int(data[3]+1)
            kv    = int(data[4]+1)
            
            counter = 10
            tu = data[counter:counter+Nctlu+ku]
            counter += (Nctlu + ku)
            
            tv = data[counter:counter+Nctlv+kv]
            counter += (Nctlv + kv)
            
            weights = data[counter:counter+Nctlu*Nctlv]
            weights = array(weights)
            if weights.all() != 1:
                print 'WARNING: Not all weight in B-spline surface are 1.\
 A NURBS surface CANNOT be replicated exactly'
            counter += Nctlu*Nctlv

            coef = zeros([Nctlu,Nctlv,3])
            for j in xrange(Nctlv):
                for i in xrange(Nctlu):
                    coef[i,j,:] = data[counter:counter +3]
                    counter+=3

            # Last we need the ranges
            range = zeros(4)
           
            range[0] = data[counter    ]
            range[1] = data[counter + 1]
            range[2] = data[counter + 2]
            range[3] = data[counter + 3]

            self.surfs.append(pySpline.surf_spline(\
                    task='create',ku=ku,kv=kv,tu=tu,tv=tv,coef=coef,\
                        range=range,no_print=self.NO_PRINT))
        # end for

        return 
  
    def _init_lifting_surface(self,*args,**kwargs):

        assert 'xsections' in kwargs and 'scale' in kwargs \
               and 'offset' in kwargs and 'Xsec' in kwargs and 'rot' in kwargs,\
               '\'xsections\', \'offset\',\'scale\' and \'X\'  and \'rot\'\
 must be specified as kwargs'

        xsections = kwargs['xsections']
        scale     = kwargs['scale']
        offset    = kwargs['offset']
        Xsec      = kwargs['Xsec']
        rot       = kwargs['rot']

        if not len(xsections)==len(scale)==offset.shape[0]:
            print 'The length of input data is inconsistent. xsections,scale,\
offset.shape[0], Xsec, rot, must all have the same size'
            print 'xsections:',len(xsections)
            print 'scale:',len(scale)
            print 'offset:',offset.shape[0]
            print 'Xsec:',Xsec.shape[0]
            print 'rot:',rot.shape[0]
            sys.exit(1)

        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'
        # end if

        if 'file_type' in kwargs:
            file_type = kwargs['file_type']
        else:
            file_type = 'xfoil'
        # end if


        if 'breaks' in kwargs:
            breaks = kwargs['breaks']
            nBreaks = len(breaks)
        else:
            nBreaks = 0
        # end if
            
        if 'nsections' in kwargs:
            nsections = kwargs['nsections']
        else: # Figure out how many sections are in each break
            nsections = zeros(nBreaks +1,'int' )
            counter = 0 
            for i in xrange(nBreaks):
                nsections[i] = breaks[i] - counter + 1
                counter = breaks[i]
            # end for
            nsections[-1] = len(xsections) - counter
        # end if

        if 'section_spacing' in kwargs:
            section_spacing = kwargs['section_spacing']
        else:
            # Generate the section spacing -> linear default
            section_spacing = []
            for i in xrange(len(nsections)):
                section_spacing.append(linspace(0,1,nsections[i]))
            # end for
        # end if

        if 'cont' in kwargs:
            cont = kwargs['cont']
        else:
            cont = [0]*nBreaks # Default is c0 contintity
        # end if 
      
       
        naf = len(xsections)
        if 'Nfoil' in kwargs:
            N = kwargs['Nfoil']
        else:
            N = 35
        # end if
        
        # ------------------------------------------------------
        # Generate the coordinates for the sections we are given 
        # ------------------------------------------------------
        X = zeros([2,N,naf,3]) #We will get two surfaces
        for i in xrange(naf):

            X_u,Y_u,X_l,Y_l = read_af(xsections[i],file_type,N)

            X[0,:,i,0] = (X_u-offset[i,0])*scale[i]
            X[0,:,i,1] = (Y_u-offset[i,1])*scale[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (X_l-offset[i,0])*scale[i]
            X[1,:,i,1] = (Y_l-offset[i,1])*scale[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    # Twist Rotation (z-Rotation)
                    X[isurf,j,i,:] = rotzV(X[isurf,j,i,:],rot[i,2]*pi/180)
                    # Dihediral Rotation (x-Rotation)
                    X[isurf,j,i,:] = rotxV(X[isurf,j,i,:],rot[i,0]*pi/180)
                    # Sweep Rotation (y-Rotation)
                    X[isurf,j,i,:] = rotyV(X[isurf,j,i,:],rot[i,1]*pi/180)
                # end ofr
            # end for

            # Finally translate according to  positions specified
            X[:,:,i,:] += Xsec[i,:]
        # end for

        # ---------------------------------------------------------------------
        # Now, we interpolate them IF we have breaks 
        # ---------------------------------------------------------------------

        self.surfs = []

        if nBreaks>0:
            tot_sec = sum(nsections)-nBreaks
            Xnew    = zeros([2,N,tot_sec,3])
            Xsecnew = zeros((tot_sec,3))
            rotnew  = zeros((tot_sec,3))
            start   = 0
            start2  = 0

            for i in xrange(nBreaks+1):
                # We have to interpolate the sectional data 
                if i == nBreaks:
                    end = naf
                else:
                    end  = breaks[i]+1
                #end if

                end2 = start2 + nsections[i]

                # We need to figure out what derivative constraints are
                # required
                
                # Create a chord line representation

                Xchord_line = array([X[0,0,start],X[0,-1,start]])
                chord_line = pySpline.linear_spline(task='interpolate',X=Xchord_line,k=2)

                for j in xrange(N): # This is for the Data points

                    if i > 0 and cont[i-1] == 1: # Do a continuity join
                        print 'cont join'
                        # Interpolate across each point in the spanwise direction
                        # Take a finite difference to get dv and normalize
                        dv = (X[0,j,start] - X[0,j,start-1])
                        dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])

                        # Now project the vector between sucessive
                        # airfoil points onto this vector                        
                        V = X[0,j,end-1]-X[0,j,start]
                        dx1 = dot(dv,V) * dv

                        # For the second vector, project the point
                        # onto the chord line of the previous section

                        # D is the vector we want
                        s,D,converged,updated = \
                            chord_line.projectPoint(X[0,j,end-1])
                        dx2 = V-D
 
                        # Now generate the line and extract the points we want
                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[0,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                   
                        Xnew[0,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                        # Interpolate across each point in the spanwise direction
                        
                        dv = (X[1,j,start]-X[1,j,start-1])
                        dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
                        V = X[0,j,end-1]-X[0,j,start]
                        dist = dv * dot(dv,V)

                        # D is the vector we want
                        s,D,converged,updated = \
                            chord_line.projectPoint(X[1,j,end-1])
                        # We already have the 'V' vector
                        V = X[1,j,end-1]-X[1,j,start]
                        dx2 = V-D

                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[1,j,start:end,:],k=4,\
                                dx1=dx1,dx2=dx2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                    else:
                            
                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[0,j,start:end,:],k=2)
                        Xnew[0,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])

                        temp_spline = pySpline.linear_spline(\
                            task='interpolate',X=X[1,j,start:end,:],k=2)
                        Xnew[1,j,start2:end2,:] = \
                            temp_spline.getValueV(section_spacing[i])
                    # end if
                # end for

                # Now we can generate and append the surfaces

                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=Xnew[0,:,start2:end2,:],\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs))
                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=Xnew[1,:,start2:end2,:],\
                            Nctlv=nsections[i],no_print=self.NO_PRINT,*args,**kwargs))

                start = end-1
                start2 = end2-1
            # end for
        
        else:  #No breaks
            tot_sec = sum(nsections)
            Xnew    = zeros([2,N,tot_sec,3])
            Xsecnew = zeros((tot_sec,3))
            rotnew  = zeros((tot_sec,3))

            for j in xrange(N):
                temp_spline = pySpline.linear_spline(task='interpolate',X=X[0,j,:,:],k=2)
                Xnew[0,j,:,:] = temp_spline.getValueV(section_spacing[0])
                temp_spline = pySpline.linear_spline(task='interpolate',X=X[1,j,:,:],k=2)
                Xnew[1,j,:,:] = temp_spline.getValueV(section_spacing[0])
            # end for
            Nctlv = nsections
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=Xnew[0],Nctlv=nsections[0],
                                                   no_print=self.NO_PRINT,*args,**kwargs))
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=Xnew[1],Nctlv=nsections[0],
                                                   no_print=self.NO_PRINT,*args,**kwargs))
        # end if

        if 'end_type' in kwargs: # The user has specified automatic tip completition
            end_type = kwargs['end_type']

            assert end_type in ['rounded','flat'],'Error: end_type must be one of \'rounded\' or \
\'flat\'. Rounded will result in a non-degenerate geometry while flat type will result in a single \
double degenerate patch at the tip'


            if end_type == 'flat':
            
                spacing = 10
                v = linspace(0,1,spacing)
                X2 = zeros((N,spacing,3))
                for j in xrange(1,N-1):
                    # Create a linear spline 
                    x1 = X[0,j,-1]
                    x2 = X[1,N-j-1,-1]

                    temp = pySpline.linear_spline(task='interpolate',\
                                                      k=2,X=array([x1,x2]))
                    X2[j,:,:] = temp.getValueV(v)

                # end for
                X2[0,:] = X[0,0,-1]
                X2[-1,:] = X[1,0,-1]
                
                self.surfs.append(pySpline.surf_spline(task='lms',ku=4,kv=4,\
                                                           X=X2,Nctlv=spacing,\
                                                           *args,**kwargs))
            elif end_type == 'rounded':
                if 'end_scale' in kwargs:
                    end_scale = kwargs['end_scale']
                else:
                    end_scale = 1
                # This code uses *some* huristic measures but generally works fairly well
                # Generate a "pinch" airfoil from the last one given

                # First determine the maximum thickness of the airfoil, since this will 
                # determine how far we need to offset it
                dist_max = 0

                for j in xrange(N):
                    dist = e_dist(X[0,j,-1],X[1,N-j-1,-1])
                    if dist > dist_max:
                        dist_max = dist
                    # end if
                # end for

                # Create a chord line representation and vector
                Xchord_line = array([X[0,0,-1],X[0,-1,-1]])
                chord_line = pySpline.linear_spline(task='interpolate',X=Xchord_line,k=2)
                chord_vec = X[0,-1,-1]-X[0,0,-1]

                # Foil Center 
                center = 0.5*(X[1,0,-1] + X[1,-1,-1])
                # Create the "airfoil" data for the pinch tip
                tip_line = zeros((N,3))
                # Create the "Average" normal vector for the section
                normal = zeros(3)
                for j in xrange(N):
                    dv_top    = (X[0,j,-1] - X[0,j,-2]) # Normal along upper surface
                    dv_bottom = (X[1,N-j-1,-1]-X[1,N-j-1,-2]) # Normal along lower surface
                    n = 0.5*(dv_top+dv_bottom) # Average
                    normal += n/sqrt(dot(n,n)) #Normalize
                # end for
                normal /= N

                for j in xrange(N):
                    tip_line[j] = (0.5*(X[0,j,-1]+X[1,N-j-1,-1])-center)*.5+center+\
                        dist_max*normal*end_scale
                # end for

                for ii in xrange(2): # up/low side loop
                    Xnew = zeros((N,6,3))
                    for j in xrange(N): # This is for the Data points
                        # Interpolate across each point in the spanwise direction
                        # Take a finite difference to get dv and normalize
                        dv = (X[ii,j,-1] - X[ii,j,-2])
                        dv /= sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])

                        # Now project the vector between sucessive
                        # airfoil points onto this vector                        
                        if ii == 0:
                            V = tip_line[j]-X[ii,j,-1]
                            s,D,converged,updated =  chord_line.projectPoint(tip_line[j])
                           
                            X_input = array([X[ii,j,-1],tip_line[j]])
                        else:
                            V = tip_line[N-j-1]-X[ii,j,-1]
                            s,D,converged,updated =  chord_line.projectPoint(tip_line[N-j-1])
                            X_input = array([X[ii,j,-1],tip_line[N-j-1]])
                        # end if
                        dx1 = dot(dv,V) * dv * end_scale
                        dx2 = V-D
                        temp_spline = pySpline.linear_spline(task='interpolate',X=X_input,
                                                             k=4,dx1=dx1,dx2=dx2)

                        Xnew[j] =  temp_spline.getValueV(linspace(0,1,6))
                    # end for
                    self.surfs.append(pySpline.surf_spline(task='lms',ku=4,kv=4,X=Xnew,
                                                           Nctlv=4, *args,**kwargs))
                # end for (ii side loop)
            # end if (tip tip if statment)
        # end if (if statment for tip type)

        self.nSurf = len(self.surfs) # And last but not least
        return

    def _init_acdt_geo(self,*args,**kwargs):

        assert 'acdt_geo' in kwargs,\
            'key word argument \'acdt_geo\' Must be specified for \
init_acdt_geo type. The user must pass an instance of a pyGeometry aircraft'

        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'

        acg = kwargs['acdt_geo']
        Components = acg._components
        ncomp = len(Components)
        counter = 0
        self.surfs = []
        # Write Aircraft Componentss
        for comp1 in xrange(ncomp):
            ncomp2 = len(Components[comp1])
            for comp2 in xrange(ncomp2):
                counter += 1
                [m,n] = Components[comp1]._components[comp2].Surface_x.shape
                X = zeros((m,n,3))
                X[:,:,0] = Components[comp1]._components[comp2].Surface_x
                X[:,:,1] = Components[comp1]._components[comp2].Surface_y
                X[:,:,2] = Components[comp1]._components[comp2].Surface_z
                self.surfs.append(pySpline.surf_spline(\
                        fit_type,ku=4,kv=4,X=X,*args,**kwargs))
            # end for
        # end for

        self.nSurf = len(self.surfs)
		
# ----------------------------------------------------------------------
#                      Edge Connection Information Functions
# ----------------------------------------------------------------------    

    def calcEdgeConnectivity(self,node_tol=1e-4,edge_tol=1e-4):

        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        
        # Calculate the NODE connectivity.
      
        node_list = [] # Physical Coordinates of the Nodes
        node_link = [] # Index in node_list for each node on surface

        for isurf in xrange(self.nSurf):
            node_link.append([])
            for inode in xrange(4): 

                node = self.surfs[isurf].getOrigValueCorner(inode)
                
                if len(node_list) == 0:
                    node_list.append(node)
                    node_link[isurf].append(inode)
                else:
                    found_it = False
                    for i in xrange(len(node_list)):
                        if e_dist(node,node_list[i]) < node_tol:
                            node_link[isurf].append(i)
                            found_it = True
                            break
                        # end if
                    # end for
                    if not found_it:
                        node_list.append(node)
                        node_link[isurf].append(i+1)
                    # end if
                # end if
            # end for
        # end for

        # Next Calculate the EDGE connectivity. 
                        
        edge_list = []
        edge_link = -1*ones((self.nSurf,4),'intc')
        edge_dir  = zeros((self.nSurf,4),'intc')
        
        for isurf in xrange(self.nSurf):
            for iedge in xrange(4):
                n1,n2 = nodesFromEdge(iedge) # nodesFromEdge in geo_utils
                n1 = node_link[isurf][n1]
                n2 = node_link[isurf][n2] 
                beg,mid_point,end = self.surfs[isurf].getOrigValuesEdge(iedge)

                if len(edge_list) == 0:
                    edge_list.append([n1,n2,mid_point,-1,0])
                    edge_link[isurf][iedge] = 0
                    edge_dir [isurf][iedge] = 1
                else:
                    found_it = False
                    for i in xrange(len(edge_list)):
                        if [n1,n2] == edge_list[i][0:2] and n1 != n2:
                            if e_dist(mid_point,edge_list[i][2]) < edge_tol:
                                edge_link[isurf][iedge] = i
                                edge_dir [isurf][iedge] = 1
                                found_it = True
                            # end if
                        elif [n2,n1] == edge_list[i][0:2] and n1 != n2:
                            if e_dist(mid_point,edge_list[i][2]) < edge_tol: # check mid_point
                                edge_link[isurf][iedge] = i
                                edge_dir[isurf][iedge] = -1
                                found_it = True
                            # end if
                        # end if
                    # end for

                    # We went all the way though the list so add it at end and return index
                    if not found_it:
                        edge_list.append([n1,n2,mid_point,-1,0])
                        edge_link[isurf][iedge] = i+1
                        edge_dir [isurf][iedge] = 1
                # end if
            # end for
        # end for
    
        # Next Calculate the Design Group Information
        dg_counter = -1
        for i in xrange(len(edge_list)):
            if edge_list[i][3] == -1: # Not set yet
                dg_counter += 1
                edge_list[i][3] = dg_counter
                self.addDGEdge(i,edge_list,edge_link)
            # end if
        # end for

        # Get Default number of ncoef in each design group

        # Lets save the stuff
        self.node_link = array(node_link)
        self.edge_list = []

        for i in xrange(len(edge_list)): # Create the edge objects
            if edge_list[i][0] == edge_list[i][1] and e_dist(edge_list[i][2],node_list[edge_list[i][0]]) < node_tol:
                # Its a degenerate edge: both node indicies are the
                # same and the midpoint is within node_tol of the end poins
                self.edge_list.append(edge(edge_list[i][0],edge_list[i][1],0,1,0,edge_list[i][3],edge_list[i][4]))
            else:
                # Its not degenerate, but may still have the same endpoints
                self.edge_list.append(edge(edge_list[i][0],edge_list[i][1],0,0,0,edge_list[i][3],edge_list[i][4]))
            # end if
        # end for
        self.edge_link = array(edge_link)
        self.edge_dir  = edge_dir
        self._setEdgeConnectivity()

        return
    
    def addDGEdge(self,i,edge_list,edge_link):
        # Find surfs with edges of 'i'
        for isurf in xrange(self.nSurf):
            for iedge in xrange(4):
                edge_num = edge_link[isurf][iedge]
                if edge_num == i:
                    if iedge in [0,1]:
                        edge_list[i][4] = self.surfs[isurf].Nctlu
                    else:
                        edge_list[i][4] = self.surfs[isurf].Nctlv

                    oppositeEdge = edge_link[isurf][flipEdge(iedge)]

                    if edge_list[oppositeEdge][3] == -1:
                        edge_list[oppositeEdge][3] = edge_list[i][3]

                        # Check if the "oppositeEdge is degenerate" since DON't recursively add for them
                        if not edge_list[oppositeEdge][0] == edge_list[oppositeEdge][1]:
                            self.addDGEdge(oppositeEdge,edge_list,edge_link)
                        # end if
                    # end if
                # end if
            # end for
        # end for
        return 

    def _setEdgeConnectivity(self):
        '''Internal function to set the global/local numbering'''
     
        # Call the calcGlobalNumbering function
        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nctlu,self.surfs[isurf].Nctlv])
        # end for

        self.Ncoef,self.g_index,self.l_index = self.calcGlobalNumbering(sizes)

        self.coef = []
        # Now Fill up the self.coef list:
        for ii in xrange(len(self.g_index)):
            isurf = self.g_index[ii][0][0]
            i = self.g_index[ii][0][1]
            j = self.g_index[ii][0][2]
            self.coef.append( self.surfs[isurf].coef[i,j])
        # end for
            
        # Finally turn self.coef into a complex array
        self.coef = array(self.coef,'D')

        # Create a PETSc vector of the global coefficients
        if USE_PETSC:
            self.petsc_coef = PETSc.Vec()
            self.petsc_coef.createSeq(3*self.Ncoef)
            self.petsc_coef[:] = self.coef.flatten().astype('d')
            self.petsc_coef.assemble()
        # end
        return

    def calcGlobalNumbering(self,sizes,surface_list=None,node_link=None,
                            edge_list=None,edge_link=None,edge_dir=None):
        '''Internal function to calculate the global/local numbering for each surface'''
#         print 'sizes:',sizes
        if surface_list == None:
            surface_list = range(0,self.nSurf) 
            

        if node_link==None and edge_list==None and edge_link == None and edge_dir == None:
            # None are specified
            node_link = self.node_link
            edge_list = self.edge_list
            edge_link = self.edge_link
            edge_dir  = self.edge_dir
        elif not node_link==None and not edge_list==None and not edge_link==None and not edge_dir==None:
            pass
        else:
            print 'Error: All of nNode,node_link,edge_list,edge_link must be given or none \
of them. If they are omited, the stored self. values are used'
            sys.exit(1)

        nNode = len(unique(node_link.flatten()))
        
        # ----------------- Start of Edge Computation ---------------------
        counter = 0
        g_index = []
        l_index = []

        assert len(sizes) == len(surface_list),'Error: The list of sizes and \
the list of surfaces must be the same length'

        # Assign unique numbers to the corners -> Corners are indexed sequentially
        node_index = arange(nNode)
        counter = len(node_index)

        edge_index = []
        for i in xrange(len(edge_list)): 
            edge_index.append([])
        # end if
        # Assign unique numbers to the edges

        for ii in xrange(len(surface_list)):
            cur_size = [sizes[ii][0],sizes[ii][0],sizes[ii][1],sizes[ii][1]]
            isurf = surface_list[ii]

            for iedge in xrange(4):
                edge = edge_link[isurf][iedge]
                    
                if edge_index[edge] == []:# Not added yet
                    if edge_list[edge].degen == 1:
                        # Get the counter value for this "node"
                        index = node_index[edge_list[edge].n1]
                        for jj in xrange(cur_size[iedge]-2):
                            edge_index[edge].append(index)
                        # end for
                    else:
                        for jj in xrange(cur_size[iedge]-2):
                            edge_index[edge].append(counter)
                            counter += 1
                        # end for
                    # end if
                # end if
            # end for
        # end for

        g_index = []  
        for i in xrange(counter): # We must add [] for the global nodes we've already deduced
            g_index.append([])
        # end for

        l_index = []
#         for i in xrange(len(edge_index)):
#             print i,edge_index[i]
        # Now actually fill everything up

        for ii in xrange(len(surface_list)):
            isurf = surface_list[ii]
            N = sizes[ii][0]
            M = sizes[ii][1]
            l_index.append(-1*ones((N,M),'intc'))
            for i in xrange(N):
                for j in xrange(M):
                    
                    type,edge,node,index = indexPosition(i,j,N,M)
                    if type == 0:           # Interior
                        l_index[ii][i,j] = counter
                        g_index.append([[isurf,i,j]])
                        counter += 1
                    elif type == 1:         # Edge
                       
                        if edge in [0,1]:
                            if edge_dir[ii][edge] == -1: # Its a reverse dir
                                cur_index = edge_index[edge_link[ii][edge]][N-i-2]
                            else:  
                                cur_index = edge_index[edge_link[ii][edge]][i-1]
                            # end if
                        else: # edge in [2,3]
                            if edge_dir[ii][edge] == -1: # Its a reverse dir
                                cur_index = edge_index[edge_link[ii][edge]][M-j-2]
                            else:  
                                cur_index = edge_index[edge_link[ii][edge]][j-1]
                            # end if
                        # end if
                        l_index[ii][i,j] = cur_index
                        g_index[cur_index].append([isurf,i,j])
                       
                            
                    else:                  # Node
                        cur_node = node_link[isurf][node]
                        l_index[ii][i,j] = node_index[cur_node]
                        g_index[node_index[cur_node]].append([isurf,i,j])
                    # end for
                # end for (j)
            # end for (i)
        # end for (ii)

        return counter,g_index,l_index

    def getReducedSetConnectivity(self,surface_list):
        '''Produce an sub-topology consisting of nNode,edge_list,node_link and edge_link for the
        surfaces contained in surface_list'''

        # First get the reduced edge_link and node_link
        new_edge_link = zeros((len(surface_list),4),'intc')
        new_edge_dir  = zeros((len(surface_list),4),'intc')
        new_node_link = zeros((len(surface_list),4),'intc')

        for ii in xrange(len(surface_list)):
            isurf = surface_list[ii]
            new_edge_link[ii] = self.edge_link[isurf]
            new_edge_dir [ii] = self.edge_dir [isurf]
            new_node_link[ii] = self.node_link[isurf]

        # end for

        # Now flatten new_edge_link and new_node_link for easier searching
        new_node_link = new_node_link.flatten()
        new_edge_link = new_edge_link.flatten()
        # Now get the unique set of nodes and edges that are left and sort

        unique_node_list = sorted(unique(new_node_link))
        unique_edge_list = sorted(unique(new_edge_link))

        # Now Re-order the nodes and edges
       
        for i in xrange(len(unique_node_list)):
            for j in xrange(len(new_node_link)):
                if new_node_link[j] == unique_node_list[i]:
                    new_node_link[j] = i
                # end if
            # end for
        # end for

        for i in xrange(len(unique_edge_list)):
            for j in xrange(len(new_edge_link)):
                if new_edge_link[j] == unique_edge_list[i]:
                    new_edge_link[j] = i
                # end if
            # end for
        # end for

        # Finally, extract the edges we need from the edge_list
        new_edge_list = []
        for i in unique_edge_list:
            new_edge_list.append(self.edge_list[i])
        # end for

        # Reshape the link arrays back to their proper size
        new_node_link = new_node_link.reshape((len(surface_list),4))
        new_edge_link = new_edge_link.reshape((len(surface_list),4))

        return new_node_link,new_edge_list,new_edge_link,new_edge_dir
    
    def printEdgeConnectivity(self,node_link=None,edge_list=None,edge_link=None,edge_dir=None):
        '''Print the Edge Connectivity to the screen'''

        if node_link == None and edge_list == None and edge_list==None and edge_dir==None:
            node_link = self.node_link
            edge_list = self.edge_list
            edge_link = self.edge_link
            edge_dir  = self.edge_dir
        # end if
        print '------------------------------------------------------------------------'
        print '%3d   %3d'%(len(self.edge_list),len(node_link))
        print 'Edge Number    |  n0  |  n1  | Cont | Degen|Intsct|  DG  | Nctl |'
        for i in xrange(len(self.edge_list)):
            edge_list[i].write_info(i,sys.stdout)
        # end for
        print 'Surface Number |  n0  |  n1  |  n2  |  n3  |  e0  |  e1  |  e2  |  e3  | dir0 | dir1 | dir2 | dir3 |'
        for i in xrange(len(node_link)):
            print '    %3d        |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d '\
                %(i,node_link[i][0],node_link[i][1],node_link[i][2],node_link[i][3],
                  edge_link[i][0],edge_link[i][1],edge_link[i][2],edge_link[i][3],
                  edge_dir[i][0],edge_dir[i][1],edge_dir[i][2],edge_dir[i][3])
        # end for
        print '------------------------------------------------------------------------'
        return

    def writeEdgeConnectivity(self,file_name):
        '''Write the full edge connectivity to a file file_name'''
        node_link = self.node_link
        edge_link = self.edge_link
        edge_list = self.edge_list
        edge_dir  = self.edge_dir
        f = open(file_name,'w')
        f.write('%3d\n'%(len(self.edge_list)))
        f.write('Edge Number    |  n0  |  n1  | Cont | Degen|Intsct|  DG  | Nctl |\n')
        for i in xrange(len(self.edge_list)):
            edge_list[i].write_info(i,f)
        # end for
        f.write('Surface Number |  n0  |  n1  |  n2  |  n3  |  e0  |  e1  |  e2  |  e3  | dir0 | dir1 | dir2 | dir3 | \n')
        for i in xrange(len(node_link)):
            f.write('    %3d        |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d \n'\
                %(i,node_link[i][0],node_link[i][1],node_link[i][2],node_link[i][3],
                  edge_link[i][0],edge_link[i][1],edge_link[i][2],edge_link[i][3],
                  edge_dir[i][0],edge_dir[i][1],edge_dir[i][2],edge_dir[i][3]))
        # end for
        f.close()
        
        return

    def readEdgeConnectivity(self,file_name):
        '''Read the full edge connectivity from a file file_name'''
        f = open(file_name,'r')

        nEdge = int(f.readline())
        self.edge_list = []
        
        f.readline() # This is the header line so ignore
        
        for i in xrange(nEdge):
            aux = string.split(f.readline(),'|')
            self.edge_list.append(edge(int(aux[1]),int(aux[2]),int(aux[3]),
                                       int(aux[4]),int(aux[5]),int(aux[6]),int(aux[7])))
        # end for

        f.readline() # This the second header line so ignore

        self.edge_link = zeros((self.nSurf,4),'intc')
        self.node_link = zeros((self.nSurf,4),'intc')
        self.edge_dir  = zeros((self.nSurf,4),'intc')
        for i in xrange(self.nSurf):
            aux = string.split(f.readline(),'|')
            
            for j in xrange(4):
                self.node_link[i][j] = int(aux[j+1])
                self.edge_link[i][j] = int(aux[j+1+4])
                self.edge_dir[i][j]  = int(aux[j+1+8])
            # end for
        # end for
                
        self.nNode = len(unique(self.node_link.flatten()))

        self._setEdgeConnectivity()

        return
    
    def propagateKnotVectors(self):

        # First get the number of design groups
        nDG = -1
        ncoef = []
        for i in xrange(len(self.edge_list)):
            if self.edge_list[i].dg > nDG:
                nDG = self.edge_list[i].dg
                ncoef.append(self.edge_list[i].Nctl)
            # end if
        # end for
        nDG += 1
        for isurf in xrange(self.nSurf):
            dg_u = self.edge_list[self.edge_link[isurf][0]].dg
            dg_v = self.edge_list[self.edge_link[isurf][2]].dg
            self.surfs[isurf].Nctlu = ncoef[dg_u]
            self.surfs[isurf].Nctlv = ncoef[dg_v]
            if self.surfs[isurf].ku < self.surfs[isurf].Nctlu:
                if self.surfs[isurf].Nctlu > 4:
                    self.surfs[isurf].ku = 4
                else:
                    self.surfs[isurf].ku = self.surfs[isurf].Nctlu
                # endif
            # end if
            if self.surfs[isurf].kv < self.surfs[isurf].Nctlv:
                if self.surfs[isurf].Nctlv > 4:
                    self.surfs[isurf].kv = 4
                else:
                    self.surfs[isurf].kv = self.surfs[isurf].Nctlv

            self.surfs[isurf]._calcKnots()
        # Now loop over the number of design groups, accumulate all
        # the knot vectors that coorspond to this dg, then merge them all
        
        for idg in xrange(nDG):
            sym = False
            knot_vectors = []
            for isurf in xrange(self.nSurf):
                # Check edge 0 and edge 2
                if self.edge_list[self.edge_link[isurf][0]].dg == idg:
                    if self.edge_dir[isurf][0] == -1 or self.edge_dir[isurf][1] == -1:
                        sym = True
                    # end if
                    knot_vectors.append(self.surfs[isurf].tu)
                # end if
                if self.edge_list[self.edge_link[isurf][2]].dg == idg:
                    if self.edge_dir[isurf][2] == -1 or self.edge_dir[isurf][3] == -1:
                        sym = True
                    # end if
                    knot_vectors.append(self.surfs[isurf].tv)
                # end if
            # end for

            # Now blend all the knot vectors

            new_knot_vec = blendKnotVectors(knot_vectors,sym)
            #print ' '
            #print 'new_knot_vec:',new_knot_vec,sym
            # And reset them all
            for isurf in xrange(self.nSurf):
                # Check edge 0 and edge 2

                if self.edge_list[self.edge_link[isurf][0]].dg == idg:
                    self.surfs[isurf].tu = new_knot_vec.copy()
                # end if
                if self.edge_list[self.edge_link[isurf][2]].dg == idg:
                    self.surfs[isurf].tv = new_knot_vec.copy()
                # end if
            # end for
        # end for
       
        if not self.NO_PRINT:
            print 'Recomputing surfaces...'
        for isurf in xrange(self.nSurf):
            self.surfs[isurf].recompute()
        # end for
        # Update the coefficients on the local surfaces
        self._setEdgeConnectivity()
        self.update()
        return


    def getSurfaceFromEdge(self,edge):
        '''Determine the surfaces and their edge_link index that points to edge iedge'''
        surfaces = []
        for isurf in xrange(self.nSurf):
            for iedge in xrange(4):
                if self.edge_link[isurf][iedge] == edge:
                    surfaces.append([isurf,iedge])
                # end if
            # end for
        # end for
        return surfaces
    
   
    def checkCoef(self):
        '''Check all surface coefficients for consistency'''
        for isurf in xrange(self.nSurf):
            print 'isurf:',isurf
            counter = self.surfs[isurf].checkCoef()
            if counter > 0:
                print '%d control points on surface %d'%(counter,isurf)
        # end for


# ----------------------------------------------------------------------
#                        Surface Fitting Functions
# ----------------------------------------------------------------------

    def fitSurfaces(self):
        '''This function does a lms fit on all the surfaces respecting
        the stitched edges as well as the continuity constraints'''

        nCtl = len(self.coef)

        sizes = []
        for isurf in xrange(self.nSurf):
            sizes.append([self.surfs[isurf].Nu,self.surfs[isurf].Nv])
        # end for
        
        # Get the Globaling number of the original data
        nPts, g_index,l_index = self.calcGlobalNumbering(sizes)
        
        nRows,nCols,dv_link = self._initJacobian(nPts)

        if not self.NO_PRINT:
            print '------------- Fitting Surfaces Globally ------------------'
            print 'nRows (Surface Points):',nRows
            print 'nCols (Degrees of Freedom):',nCols

        if USE_PETSC:
            pts = PETSc.Vec().createSeq(nRows)
            temp= PETSc.Vec().createSeq(nRows)
            X = PETSc.Vec().createSeq(nCols)
            X_cur = PETSc.Vec().createSeq(nCols)
        else:
            pts = zeros(nRows) 
            temp = None
            X = zeros(nCols)
            X_cur = zeros(nCols)
        # end if 
      
        # Fill up the 'X' with the best curent solution guess
        for i in xrange(len(dv_link)):
            if len(dv_link[i][0]) == 1: # Its regular
                X[dv_link[i][0][0]:dv_link[i][0][0]+3] = self.coef[i].astype('d')
            else:
                X[dv_link[i][0][0]] = 0.5
                dv_index = dv_link[i][0][0]
                n1_index = dv_link[i][0][1] # node one side of constrined node
                n2_index = dv_link[i][0][2] # node other side of constrained node
                self.coef[i] = self.coef[n1_index]*(1-X[dv_index]) + X[dv_index]*self.coef[n2_index]
            # end if
        # end for
        
        if USE_PETSC:
            X.copy(X_cur)
        else:
            X_cur = X.copy()
        # end if



        # Now Fill up the RHS point list
        for ii in xrange(len(g_index)):
            isurf = g_index[ii][0][0]
            i = g_index[ii][0][1]
            j = g_index[ii][0][2]
            pts[3*ii:3*ii+3] = self.surfs[isurf].X[i,j]
        # end for
        rhs = pts
        if not self.NO_PRINT:
            print 'LMS solving...'
        nIter = 6
        for iter in xrange(nIter):
            # Assemble the Jacobian
            nRows,nCols,dv_link = self._initJacobian(nPts)
            for ii in xrange(nPts):
                surfID = g_index[ii][0][0]
                i      = g_index[ii][0][1]
                j      = g_index[ii][0][2]

                u = self.surfs[surfID].u[i]
                v = self.surfs[surfID].v[j]

                ku = self.surfs[surfID].ku
                kv = self.surfs[surfID].kv

                ileftu, mflagu = self.surfs[surfID].pyspline.intrv(\
                    self.surfs[surfID].tu,u,1)
                ileftv, mflagv = self.surfs[surfID].pyspline.intrv(\
                    self.surfs[surfID].tv,v,1)

                if mflagu == 0: # Its Inside so everything is ok
                    u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
                if mflagu == 1: # Its at the right end so just need last one
                    u_list = [ileftu-ku-1]

                if mflagv == 0: # Its Inside so everything is ok
                    v_list = [ileftv-kv,ileftv-kv+1,ileftv-kv+2,ileftv-kv+3]
                if mflagv == 1: # Its at the right end so just need last one
                    v_list = [ileftv-kv-1]

                for iii in xrange(len(u_list)):
                    for jjj in xrange(len(v_list)):
                        # Should we need a += here??? I don't think so...
                        x = self.surfs[surfID].calcPtDeriv(\
                            u,v,u_list[iii],v_list[jjj])

                        # X is the derivative of the physical point at parametric location u,v
                        # by control point u_list[iii],v_list[jjj]

                        global_index = self.l_index[surfID][u_list[iii],v_list[jjj]]
                        if len(dv_link[global_index][0]) == 1:
                            dv_index = dv_link[global_index][0][0]
                            self._addJacobianValue(3*ii    ,dv_index    ,x)
                            self._addJacobianValue(3*ii + 1,dv_index + 1,x)
                            self._addJacobianValue(3*ii + 2,dv_index + 2,x)
                        else: # its a constrained one
                            dv_index = dv_link[global_index][0][0]
                            n1_index = dv_link[global_index][0][1] # node one side of constrined node
                            n2_index = dv_link[global_index][0][2] # node other side of constrained node
                          #   print '1:',dv_index
                            dv1 = dv_link[n1_index][0][0]
                            dv2 = dv_link[n2_index][0][0]
                            
                            dcoefds = -self.coef[n1_index] + self.coef[n2_index]
                            self._addJacobianValue(3*ii    ,dv_index,x*dcoefds[0])
                            self._addJacobianValue(3*ii + 1,dv_index,x*dcoefds[1])
                            self._addJacobianValue(3*ii + 2,dv_index,x*dcoefds[2])

                            # We also need to add the dependance of the other two nodes as well
                            #print '1:',global_index
                            dv_index = dv_link[n1_index][0][0]
                            #print '2:',n1_index,dv_index
                            self._addJacobianValue(3*ii    ,dv_index  ,(1-X[dv_index])*x)
                            self._addJacobianValue(3*ii + 1,dv_index+1,(1-X[dv_index])*x)
                            self._addJacobianValue(3*ii + 2,dv_index+2,(1-X[dv_index])*x)
                            
                            dv_index = dv_link[n2_index][0][0]
                            #print '3:',n2_index,dv_index
                            self._addJacobianValue(3*ii    ,dv_index  ,X[dv_index]*x)
                            self._addJacobianValue(3*ii + 1,dv_index+1,X[dv_index]*x)
                            self._addJacobianValue(3*ii + 2,dv_index+2,X[dv_index]*x)

                        # end if
                    # end for
                # end for
            # end for 
            if iter == 0:
                if USE_PETSC:
                    self.J.assemblyBegin()
                    self.J.assemblyEnd()
                    self.J.mult(X,temp)
                    rhs = rhs - temp
                else:
                    rhs -= dot(self.J,X)
                # end if
            # end if
            rhs,X,X_cur = self._solve(X,X_cur,rhs,temp,dv_link,iter)
        # end for (iter)
        return

    def _addJacobianValue(self,i,j,value):
        if USE_PETSC: 
            self.J.setValue(i,j,value,PETSc.InsertMode.ADD_VALUES)
        else:
            self.J[i,j] += value
        # end if

    def _solve(self,X,X_cur,rhs,temp,dv_link,iter):
        '''Solve for the control points'''
        

        if USE_PETSC:

            self.J.assemblyBegin()
            self.J.assemblyEnd()

            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.getPC().setType('none')
            ksp.setType('lsqr')
            #ksp.setInitialGuessNonzero(True)

            print 'Iteration   Residual'
            def monitor(ksp, its, rnorm):
                if mod(its,50) == 0:
                    print '%5d      %20.15g'%(its,rnorm)

            ksp.setMonitor(monitor)
            ksp.setTolerances(rtol=1e-15, atol=1e-15, divtol=100, max_it=250)

            ksp.setOperators(self.J)
            ksp.solve(rhs, X)
            self.J.mult(X,temp)
            rhs = rhs - temp

        else:

            X = lstsq(self.J,rhs)[0]
            rhs -= dot(self.J,X)
            print 'rms:',sqrt(dot(rhs,rhs))

        # end if
        scale = 1
        X_cur = X_cur + X/scale

        for icoef in xrange(len(self.coef)):
            if len(dv_link[icoef][0]) == 1:
                dv_index = dv_link[icoef][0][0]
                self.coef[icoef,0] = (X_cur[dv_index + 0])
                self.coef[icoef,1] = (X_cur[dv_index + 1])
                self.coef[icoef,2] = (X_cur[dv_index + 2])
            # end if
        for icoef in xrange(len(self.coef)):
            if len(dv_link[icoef][0]) != 1:
                dv_index = dv_link[icoef][0][0]
                n1_index = dv_link[icoef][0][1] # node one side of constrined node
                n2_index = dv_link[icoef][0][2] # node other side of constrained node
                dv1 = dv_link[n1_index][0][0]
                dv2 = dv_link[n2_index][0][0]
                #print 'Value1:',X_cur[dv_index]

                update0 = X[dv_index]/scale
                value = update0
                for i in xrange(25):
                    if abs(value) > 0.1:
                        value /= 2
                    else:
                        break
                
                # end for
                # We've already added update---but we really want to add value instread
                #print 'update0,value:',update0,value
                X_cur[dv_index] = X_cur[dv_index] - update0 +value
                value = X_cur[dv_index]
                #value = .5
                #X_cur[dv_index] = .5
                print 'Value2:',X_cur[dv_index]
                
                self.coef[icoef] = (1-value)*self.coef[n1_index] + value*(self.coef[n2_index])
              
            # end if
        # end for

        return rhs,X,X_cur

    def _initJacobian(self,Npt):
        
        '''Initialize the Jacobian either with PETSc or with Numpy for use
        with LAPACK'''
        
        dv_link = [-1]*len(self.coef)
        dv_counter = 0
        for isurf in xrange(self.nSurf):
            Nctlu = self.surfs[isurf].Nctlu
            Nctlv = self.surfs[isurf].Nctlv
            for i in xrange(Nctlu):
                for j in xrange(Nctlv):
                    type,edge,node,index = indexPosition(i,j,Nctlu,Nctlv)
                    if type == 0: # Interior
                        dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
                        dv_counter += 3
                    elif type == 1: # Edge
                        if dv_link[self.l_index[isurf][i,j]] ==-1: # Its isn't set yet
                            # Now determine if its on a continuity edge
                            if self.edge_list[self.edge_link[isurf][edge]].cont == 1: #its continuous
                                iedge = self.edge_link[isurf][edge] # index of edge of interest
                                surfaces = self.getSurfaceFromEdge(iedge) # Two surfaces we want

                                surf0 = surfaces[0][0] # First surface on this edge
                                edge0 = surfaces[0][1] # Edge of surface on this edge                           
                                surf1 = surfaces[1][0] # Second surface on this edge
                                edge1 = surfaces[1][1] # Edge of second surface on this edge

                                tindA,indB = self._getTwoIndiciesOnEdge(
                                    self.l_index[surf0],index,edge0,self.edge_dir[surf0])

                                tindA,indC = self._getTwoIndiciesOnEdge(
                                    self.l_index[surf1],index,edge1,self.edge_dir[surf1])

                                # indB and indC are the global indicies of the two control 
                                # points on either side of this node on the edge

                                dv_link[self.l_index[isurf][i,j]] = [[dv_counter,indB,indC]]
                                dv_counter += 1
                            else: # Just add normally
                                dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
                                dv_counter += 3
                            # end if
                        # end if
                    elif type == 2: # Corner
                        if dv_link[self.l_index[isurf][i,j]] == -1: # Its not set yet
                            # Check both possible edges
                            edge1,edge2,index1,index2 = edgesFromNodeIndex(node,Nctlu,Nctlv)
                            edges= [edge1,edge2]
                            indices = [index1,index2]
                            dv_link[self.l_index[isurf][i,j]] = []
                            for ii in xrange(2):
                                if self.edge_list[self.edge_link[isurf][edges[ii]]].cont == 1:
                                    iedge = self.edge_link[isurf][edges[ii]] # index of edge of interest
                                    surfaces = self.getSurfaceFromEdge(iedge) # Two surfaces we want
                                    surf0 = surfaces[0][0] # First surface on this edge
                                    edge0 = surfaces[0][1] # Edge of surface on this edge                           
                                    surf1 = surfaces[1][0] # Second surface on this edge
                                    edge1 = surfaces[1][1] # Edge of second surface on this edge
                                    
                                    tindA,indB = self._getTwoIndiciesOnEdge(
                                        self.l_index[surf0],indices[ii],edge0,self.edge_dir[surf0])

                                    tindA,indC = self._getTwoIndiciesOnEdge(
                                        self.l_index[surf1],indices[ii],edge1,self.edge_dir[surf1])

                                    # indB and indC are the global indicies of the two control 
                                    # points on either side of this node on the edge
                                    dv_link[self.l_index[isurf][i,j]].append([dv_counter,indB,indC])
                                    dv_counter += 1

                                # end if
                            # end for
                            # If its STILL not set there's no continutiy
                            if dv_link[self.l_index[isurf][i,j]] == []: # Need this check again
                                dv_link[self.l_index[isurf][i,j]] = [[dv_counter]]
                                dv_counter += 3
                            # end if
                    # end if (pt type)
                # end for (Nctlv loop)
            # end for (Nctlu loop)
        # end for (isurf looop)
                                
        nRows = Npt*3
        nCols = dv_counter

        if USE_PETSC:
            self.J = PETSc.Mat()
            # We know the row filling factor: 16*3 (4 for ku by 4 for
            # kv and 3 spatial)
            if PETSC_MAJOR_VERSION == 1:
                self.J.createAIJ([nRows,nCols],nnz=16*3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                self.J.createSeqAIJ([nRows,nCols],nz=16*3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            self.J = zeros((nRows,nCols))
        # end if
        return nRows,nCols,dv_link


    def _getTwoIndiciesOnEdge(self,interpolant,index,edge,edge_dir):
        '''for a given interpolat matrix, get the two values in interpolant
        that coorspond to \'index\' along \'edge\'. The direction is
        accounted for by edge_dir'''
        N = interpolant.shape[0]
        M = interpolant.shape[1]
        if edge == 0:
            if edge_dir[0] == 1:
                return interpolant[index,0],interpolant[index,1]
            else:
                return interpolant[N-index-1,0],interpolant[N-index-1,1]
        elif edge == 1:
            if edge_dir[1] == 1:
                return interpolant[index,-1],interpolant[index,-2]
            else:
                return interpolant[N-index-1,-1],interpolant[N-index-1,-2]
        elif edge == 2:
            if edge_dir[2] == 1:
                return interpolant[0,index],interpolant[1,index]
            else:
                return interpolant[0,M-index-1],interpolant[1,M-index-1]
        elif edge == 3:
            if edge_dir[3] == 1:
                return interpolant[-1,index],interpolant[-2,index]
            else:
                return interpolant[-1,M-index-1],interpolant[-2,M-index-1]
# ----------------------------------------------------------------------
#                Reference Axis Handling
# ----------------------------------------------------------------------

    def addRefAxis(self,surf_ids,X,rot,nrefsecs=None,spacing=None,\
                       point_select=None):
            '''Add surf_ids surfacs to a new reference axis defined by X and
             rot with nsection values'''
            if not self.NO_PRINT:
                print 'Adding ref axis...'
            # A couple of things can happen here: 
            # 1. nsections < len(X)
            #    -> We do a LMS fit on the ref axis (subsample)
            # 2. nsection == len(X)
            #    -> We can make the ref axis as is
            # 3. nsection < len(X)
            #    -> We reinterpolate before making the ref axis (supersample)


            print 'surf_ids:',surf_ids
            if nrefsecs == None:
                nrefsecs = X.shape[0]

            if nrefsecs < X.shape[0]:

                # Do the lms fit
                x = pySpline.linear_spline(task='lms',X=X,\
                                                  nCtl=nrefsecs,k=2)
                s = x.s
                rotxs = pySpline.linear_spline(task='lms',s=s,X=rot[:,0],\
                                                   nCtl=nrefsecs,k=2)
                rotys = pySpline.linear_spline(task='lms',s=s,X=rot[:,1],\
                                                   nCtl=nrefsecs,k=2)
                rotzs = pySpline.linear_spline(task='lms',s=s,X=rot[:,2],\
                                                   nCtl=nrefsecs,k=2)

                if not spacing == None:
                    spacing = linspace(0,1,nrefsecs)
                    
                Xnew = x.getValue(spacing)
                rotnew = zeros((nrefsecs,3))
                rotnew[:,0] = rotxs.getValueV(spacing)
                rotnew[:,1] = rotys.getValueV(spacing)
                rotnew[:,2] = rotzs.getValueV(spacing)

                
            elif nrefsecs == X.shape[0]:
                Xnew = X
                rotnew = rot

            else: #nrefsecs > X.shape
                if spacing == None:
                    spacing = linspace(0,1,nrefsecs)
                # end if

                # Do the interpolate fit
                x = pySpline.linear_spline(task='interpolate',X=X,k=2)
                s = x.s
                rotxs = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,0],nCtl=nrefsecs,k=2)
                rotys = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,1],nCtl=nrefsecs,k=2)
                rotzs = pySpline.linear_spline(\
                    task='interpolate',s=s,X=rot[:,2],nCtl=nrefsecs,k=2)

                if not spacing == None:
                    spacing = linspace(0,1,nrefsecs)
                    
                Xnew = x.getValueV(spacing)
                rotnew = zeros((nrefsecs,3))
                rotnew[:,0] = rotxs.getValueV(spacing)
                rotnew[:,1] = rotys.getValueV(spacing)
                rotnew[:,2] = rotzs.getValueV(spacing)
                
            # end if

            # create the ref axis:
            ra = ref_axis(Xnew,rotnew)

            coef_list = []
            if point_select == None: # It is was not defined -> Assume full surface
                for isurf in surf_ids:
                    for i in xrange(self.surfs[isurf].Nctlu):
                        for j in xrange(self.surfs[isurf].Nctlv):
                            coef_list.append(self.l_index[isurf][i,j])
                        # end for
                    # end for
                # end for
            # end if

            else:   # We have a point selection class passed in
                for isurf in surf_ids:
                    coef_list = point_select.getControlPoints(\
                        self.surfs[isurf],isurf,coef_list,self.l_index)
                # end for
            # end if

            # Now parse out duplicates and sort
            coef_list = unique(coef_list) #unique is in geo_utils
            coef_list.sort()
            N = len(coef_list)

            # For each surface affected, produce the s attachment
            # point list

            attachment_points = []
            types = []
            for isurf in xrange(self.nSurf):
                if isurf in surf_ids: # If this one is in the list
                    index = self.getL_surfs_index(isurf)
                    if not index == None:
                        surface_list = self.l_surfs[index]
                    else:
                        surface_list = []

                    s,type = self.getRefAxisConnection(ra,isurf,surface_list)

                    attachment_points.append(s)
                    types.append(type)
                else:
                    attachment_points.append([])
                    types.append([])
                # end if
            # end for
    
            for icoef in xrange(len(coef_list)):
                for jj in xrange(len(self.g_index[coef_list[icoef]])):
                    surfID = self.g_index[coef_list[icoef]][jj][0]
                    i = self.g_index[coef_list[icoef]][jj][1]
                    j = self.g_index[coef_list[icoef]][jj][2]

                    if surfID in surf_ids:
                        break
                    # end if
                # end for

                type = types[surfID]

                if type == 0: # Along u
                    s = attachment_points[surfID][i]
                else:
                    s = attachment_points[surfID][j]
                # end if
            
                D = self.coef[coef_list[icoef]] - ra.xs.getValue(s)
                M = ra.getRotMatrixGlobalToLocal(s)
                D = dot(M,D) #Rotate to local frame
                ra.links_s.append(s)
                ra.links_x.append(D)
            # end for
            ra.coef_list = coef_list
            ra.surf_ids  = surf_ids
            # Add the reference axis to the pyGeo list
            self.ref_axis.append(ra)
            
    def addRefAxisCon(self,axis1,axis2,con_type):
        '''Add a reference axis connection to the connection list'''
        
        # Attach axis2 to axis1 
        # Find out the POSITION and DISTANCE on
        # axis1 that axis2 will be attached
        
        s,D,converged,update = self.ref_axis[axis1].xs.projectPoint(\
            self.ref_axis[axis2].xs.getValue(0))

        M = self.ref_axis[axis1].getRotMatrixGlobalToLocal(s)
        D = dot(M,D)

        self.ref_axis[axis2].base_point_s = s
        self.ref_axis[axis2].base_point_D = D
        self.ref_axis[axis2].con_type = con_type
        if con_type == 'full':
            assert self.ref_axis[axis2].N == 2, 'Full reference axis connection \
is only available for reference axis with 2 points. A typical usage is for \
a flap hinge line'
            
            s,D,converged,update = self.ref_axis[axis1].xs.projectPoint(\
                self.ref_axis[axis2].xs.getValue(1.0))

            M = self.ref_axis[axis1].getRotMatrixGlobalToLocal(s)
            D = dot(M,D)

            self.ref_axis[axis2].end_point_s = s
            self.ref_axis[axis2].end_point_D = D
            
        # end if
            
        self.ref_axis_con.append([axis1,axis2,con_type])

        return

    def getL_surfs_index(self,isurf):
        '''Return the index of l_surfs for surface isurf'''
        for i in xrange(len(self.l_surfs)):
            for j in xrange(len(self.l_surfs[i])):
                if isurf == self.l_surfs[i][j]:
                    return i
                # end if
            # end for
        # end for
        
        return None

    def getRefAxisConnection(self,ref_axis,isurf,surface_list):
        '''Determine the primary orientation of a reference axis, ref_axis on
        surface, surface. The function returns a vector of length Nctlu or
        Nctlv whcih contains the s-positions where lines of constant u or
        v should connect to the ref axis'''


        # We need to deduce along which direction (u or v) the
        # reference axis is directed.  First estimate Over what
        # portion the surface and ref axis coinside

        # Take N Normal Vectors
        full_surface_list = [isurf]
        for extra_surf in surface_list:
            full_surface_list.append(extra_surf)
        # end for
            
        full_surface_list = unique(full_surface_list)
        
        types = []
     
        for surfid in full_surface_list:
            dir_type = directionAlongSurface(self.surfs[surfid],ref_axis.xs)
            if dir_type == 0 or dir_type == 1: # u type regarless of direction
                types.append(0)
            else:
                types.append(1)
            # end if

            if surfid == isurf:
                isurf_dir  = types[-1]
            # end if

        # end for
        
        if isurf_dir == 1: #along v of isurf
            if not self.NO_PRINT:
                print 'Reference axis is oriented along v on \
surface %d'%(isurf)
            Nctlv = self.surfs[isurf].Nctlv
            Nctlu = self.surfs[isurf].Nctlu
            s = zeros(Nctlv)
            for j in xrange(Nctlv):
                # Get ALL coefficients from surfaces in full_surface_list
                coef = []
                for jj in xrange(len(full_surface_list)):
                    if types[jj] == 0:
                        coef.append(self.surfs[full_surface_list[jj]].coef[j,:])
                    else:
                        coef.append(self.surfs[full_surface_list[jj]].coef[:,j])
                    # end if
                # end for

                X = array(coef).reshape(Nctlu*len(full_surface_list),3)
             
                temp = pySpline.linear_spline(
                    task='lms',X=X,k=2,Nctl=2)
                
                s1,s2,d,converged  = ref_axis.xs.minDistance(temp)
                s[j] = s1
            # end for

            return s,1
        else:
            if not self.NO_PRINT:
                print 'Reference axis is oriented along u on \
surface %d'%(isurf)
            Nctlu = self.surfs[isurf].Nctlu
            Nctlv = self.surfs[isurf].Nctlv
            s = zeros(Nctlu)
            for i in xrange(Nctlu):
                # Get ALL coefficients from surfaces in full_surface_list
                coef = []
                for jj in xrange(len(full_surface_list)):
                    if types[jj] == 1:
                        coef.append(self.surfs[full_surface_list[jj]].coef[:,i])
                    else:
                        coef.append(self.surfs[full_surface_list[jj]].coef[i,:])
                    # end if
                # end for
                
                X = array(coef).reshape(Nctlv*len(full_surface_list),3)
                temp = pySpline.linear_spline(
                    task='lms',X=X,k=2,Nctl=2)

                s1,s2,d,converged  = ref_axis.xs.minDistance(temp)
                s[i] = s1
            # end for
           
            return s,0

# ----------------------------------------------------------------------
#                Update and Derivative Functions
# ----------------------------------------------------------------------

    def _updateCoef(self,local=True):
        '''update the entire pyGeo Object'''
        
        # First, update the reference axis info from the design variables
        for i in xrange(len(self.DV_listGlobal)):
            # Call the each design variable with the ref axis list
            self.ref_axis = self.DV_listGlobal[i](self.ref_axis)
        # end for

        # Second, update the end_point base_point on the ref_axis:
        
        if len(self.ref_axis_con)> 0:
            for i in xrange(len(self.ref_axis_con)):
                axis1 = self.ref_axis_con[i][0]
                axis2 = self.ref_axis_con[i][1]

                self.ref_axis[axis1].update()
                s = self.ref_axis[axis2].base_point_s
                D = self.ref_axis[axis2].base_point_D
                M = self.ref_axis[axis1].getRotMatrixLocalToGlobal(s)
                D = dot(M,D)

                X0 = self.ref_axis[axis1].xs.getValue(s)

                self.ref_axis[axis2].base_point = X0 + \
                    D*self.ref_axis[axis1].scales(s)

                if self.ref_axis[axis2].con_type == 'full':
                    s = self.ref_axis[axis2].end_point_s
                    D = self.ref_axis[axis2].end_point_D
                    M = self.ref_axis[axis1].getRotMatrixLocalToGlobal(s)
                    D = dot(M,D)

                    X0 = self.ref_axis[axis1].xs.getValue(s)

                    self.ref_axis[axis2].end_point = X0 +\
                        D*self.ref_axis[axis1].scales(s)
                # end if
                self.ref_axis[axis2].update()
        else:
            for r in xrange(len(self.ref_axis)):
                self.ref_axis[r].update()
            # end for
       
        # Third, update the coefficients (from global DV changes)
        for r in xrange(len(self.ref_axis)):
            # Call the fortran function
            ra = self.ref_axis[r]
            rot = zeros((len(ra.xs.s),3),'D')
            rot[:,0] = ra.rotxs.coef
            rot[:,1] = ra.rotys.coef
            rot[:,2] = ra.rotzs.coef
                        
            #coef = getcoef(type,s_pos,links,coef,indicies,s,t,x,rot,scale)
            if ra.con_type == 'full':
                self.coef = pySpline.pyspline_cs.getcomplexcoef(\
                1,ra.links_s,ra.links_x,self.coef,ra.coef_list,\
                        ra.xs.s,ra.xs.t,ra.xs.coef,rot,ra.scales.coef)
            else:
                self.coef = pySpline.pyspline_cs.getcomplexcoef(\
                    0,ra.links_s,ra.links_x,self.coef,ra.coef_list,\
                        ra.xs.s,ra.xs.t,ra.xs.coef,rot,ra.scales.coef)
            # end if
#---------------- PYTHON IMPLEMENTATION  ------------------
           #  for i in xrange(len(ra.links_s)):
#                 base_point = ra.xs.getValue(ra.links_s[i])
#                 D = ra.links_x[i]
#                 M = ra.getRotMatrixLocalToGlobal(ra.links_s[i])
#                 D = dot(M,D)
#                 coef[ra.coef_list[i]] = base_point + D*ra.scales(s)
#             # end for
# ---------------------------------------------------------
        # end for

        
        if local:
            # fourth, update the coefficients (from normal DV changes)        
            for i in xrange(len(self.DV_listNormal)):
                surface = self.surfs[self.DV_listNormal[i].surface_id]
                self.coef = self.DV_listNormal[i](surface,self.coef)
            # end for

            # fifth: update the coefficient from local DV changes
            
            for i in xrange(len(self.DV_listLocal)):
                self.coef = self.DV_listLocal[i](self.coef)
            # end for
        # end if

        return
         
    def update(self):
        '''Run the update coefficients command and then set the control
        points'''
        self._updateCoef(local=True)

        # Update the values in PETSc
        if USE_PETSC:
            self.petsc_coef[:] = self.coef.flatten().astype('d')
            self.petsc_coef.assemble()
        # end
            
        self._updateSurfaceCoef()
        return

    def _updateSurfaceCoef(self):
        '''Copy the pyGeo list of control points back to the surfaces'''
        for ii in xrange(len(self.coef)):
            for jj in xrange(len(self.g_index[ii])):
                isurf = self.g_index[ii][jj][0]
                i     = self.g_index[ii][jj][1]
                j     = self.g_index[ii][jj][2]
                self.surfs[isurf].coef[i,j] = self.coef[ii].astype('d')
            # end for
        # end for
        return

    def getSizes( self ):
        '''
        Get the sizes:
        - The number of global design variables
        - The number of normal design variables
        - The number of local design variables
        - The number of control points
        '''
        
        # Initialize the jacobian
        # Calculate the size Ncoef x Ndesign Variables
        Nctl = len(self.coef)

        # Calculate the Number of Design Variables:
        N = 0
        for i in xrange(len(self.DV_listGlobal)): #Global Variables
            if self.DV_listGlobal[i].useit:
                N += self.DV_listGlobal[i].nVal
            # end if
        # end for
            
        NdvGlobal = N
        
        for i in xrange(len(self.DV_listNormal)): # Normal Variables
            N += self.DV_listLocal[i].nVal
        # end for
                
        NdvNormal = N-NdvGlobal

        for i in xrange(len(self.DV_listLocal)): # Local Variables
            N += self.DV_listLocal[i].nVal*3
        # end for
                
        NdvLocal = N-(NdvNormal+NdvGlobal)

        return NdvGlobal, NdvNormal, NdvLocal, Nctl


    def _initdCoefdx( self ):
        '''
        Allocate the space for dCoefdx and perform some setup        
        '''

        NdvGlobal, NdvNormal, NdvLocal, Nctl = self.getSizes()
        Ndv = NdvGlobal + NdvNormal + NdvLocal
        
        if USE_PETSC:
            dCoefdx = PETSc.Mat()
            
            # We know the row filling factor: Its (exactly) nGlobal + 3            
            if PETSC_MAJOR_VERSION == 1:
                dCoefdx.createAIJ([Nctl*3,Ndv],nnz=NdvGlobal+3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                dCoefdx.createSeqAIJ([Nctl*3,Ndv],nz=NdvGlobal+3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            dCoefdx = zeros((Nctl*3,Ndv))
        # end if

        return dCoefdx
        
    def calcCtlDeriv(self):

        '''This function runs the complex step method over the design variable
        and generates a (sparse) jacobian of the control pt
        derivatives wrt to the design variables'''

        if self.dCoefdx == None:
            self.dCoefdx = self._initdCoefdx()
        # end
   
        h = 1.0e-40j
        col_counter = 0
        for idv in xrange(len(self.DV_listGlobal)): # This is the Master CS Loop
            if self.DV_listGlobal[idv].useit:
                nVal = self.DV_listGlobal[idv].nVal

                for jj in xrange(nVal):
                    if nVal == 1:
                        self.DV_listGlobal[idv].value += h
                    else:
                        self.DV_listGlobal[idv].value[jj] += h
                    # end if

                    # Now get the updated coefficients and set the column
                    self._updateCoef(local=False)
                    self.dCoefdx[:,col_counter] = imag(self.coef.flatten())/1e-40
                    col_counter += 1    # Increment Column Counter

                    # Reset Design Variable Peturbation
                    if nVal == 1:
                        self.DV_listGlobal[idv].value -= h
                    else:
                        self.DV_listGlobal[idv].value[jj] -= h
                    # end if
                # end for (nval loop)
            # end if (useit)
        # end for (outer design variable loop)
        
        # The next step is go to over all the NORMAL and LOCAL variables,
        # compute the surface normal
        
        for idv in xrange(len(self.DV_listNormal)): 
            surface = self.surfs[self.DV_listNormal[idv].surface_id]
            normals = self.DV_listNormal[idv].getNormals(\
                surface,self.coef.astype('d'))

            # Normals is the length of local dv on this surface
            for i in xrange(self.DV_listNormal[idv].nVal):
                index = 3*self.DV_listNormal[idv].coef_list[i]
                self.dCoefdx[index:index+3,col_counter] = normals[i,:]
                col_counter += 1
            # end for
        # end for

        for idv in xrange(len(self.DV_listLocal)):
            for i in xrange(self.DV_listLocal[idv].nVal):
                for j in xrange(3):
                    index = 3*self.DV_listLocal[idv].coef_list[i]
                    self.dCoefdx[index+j,col_counter] = 1.0
                    col_counter += 1
                # end for
            # end for
        # end for
            
        if USE_PETSC:
            self.dCoefdx.assemblyBegin()
            self.dCoefdx.assemblyEnd()
        # end if 

        return

    def compute_dPtdx( self ):
        '''
        Compute the product of the derivative of the surface points w.r.t.
        the control points and the derivative of the control points w.r.t.
        the design variables. This gives the derivative of the surface points
        w.r.t. the design variables: a Jacobian matrix.
        '''
        
        # Now Do the Try the matrix multiplication
        
        # Now Do the matrix multiplication
        if USE_PETSC:
            if self.dPtdCoef:
                if self.dPtdx == None:
                    self.dPtdx = PETSc.Mat()
                # end
                self.dPtdCoef.matMult(self.dCoefdx,result=self.dPtdx)
            # end
        else:
            if self.dPtdCoef:
                self.dPtdx = dot(self.dPtdCoef,self.dCoefdx)
            # end
        # end if

        return 

    def getSurfacePoints(self,patchID,uv):

        '''Function to return ALL surface points'''

        N = len(patchID)
        coordinates = zeros((N,3))
        for i in xrange(N):
            coordinates[i] = self.surfs[patchID[i]].getValue(uv[i][0],uv[i][1])

        return coordinates.flatten()

    def addGeoDVNormal(self,dv_name,lower,upper,surf=None,point_select=None,\
                           overwrite=False):

        '''Add a normal local design variable group.'''

        if surf == None:
            print 'Error: A surface must be specified with surf = <surf_id>'
            sys.exit(1)
        # end if

        coef_list = []
        if point_select == None:
            counter = 0
            # Assume all control points on surface are to be used
            for i in xrange(self.surfs[surf].Nctlu):
                for j in xrange(self.surfs[surf].Nctlv):
                    coef_list.append(self.l_index[surf][i,j])
                # end for
            # end for
        else:
            # Use the point select class to get the indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],isurf,coef_list,l_index)
        # end if
        
        # Now, we have the list of the conrol points that we would
        # LIKE to add to this dv group. However, some may already be
        # specified in other normal of local dv groups. 

        if overwrite:
            # Loop over ALL normal and local group and force them to
            # remove all dv in coef_list

            for idv in xrange(len(self.DV_listNormal)):
                self.DV_listNormal[idv].removeCoef(coef_list)
            # end for
            
            for idv in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[idv].removeCoef(coef_list)
        else:
            # We need to (possibly) remove coef from THIS list since
            # they already exist on other dvlocals or dvnormals
           
            new_list = copy.copy(coef_list)
            for i in xrange(len(coef_list)):

                for idv in xrange(len(self.DV_listNormal)):
                    if coef_list[i] in self.DV_listNormal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
                for idv in xrange(len(self.DV_listLocal)):
                    if coef_list[i] in self.DV_listLocal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
            # end for
            coef_list = new_list
        # end if

        self.DV_listNormal.append(geoDVNormal(\
                dv_name,lower,upper,surf,coef_list,self.g_index))
        self.DV_namesNormal[dv_name] = len(self.DV_listLocal)-1
        
        return

    def addGeoDVLocal(self,dv_name,lower,upper,surf=None,point_select=None,\
                          overwrite=False):

        '''Add a general local design variable group.'''

        if surf == None:
            print 'Error: A surface must be specified with surf = <surf_id>'
            sys.exit(1)
        # end if

        coef_list = []
        if point_select == None:
            counter = 0
            # Assume all control points on surface are to be used
            for i in xrange(self.surfs[surf].Nctlu):
                for j in xrange(self.surfs[surf].Nctlv):
                    coef_list.append(self.l_index[surf][i,j])
                # end for
            # end for
        else:
            # Use the bounding box to find the appropriate indicies
            coef_list = point_select.getControlPoints(\
                self.surfs[surf],isurf,coef_list,l_index)
        # end if
        
        # Now, we have the list of the conrol points that we would
        # LIKE to add to this dv group. However, some may already be
        # specified in other normal or local dv groups. 

        if overwrite:
            # Loop over ALL normal and local group and force them to
            # remove all dv in coef_list

            for idv in xrange(len(self.DV_listNormal)):
                self.DV_listNormal[idv].removeCoef(coef_list)
            # end for
            
            for idv in xrange(len(self.DV_listLocal)):
                self.DV_listLocal[idv].removeCoef(coef_list)
        else:
            # We need to (possibly) remove coef from THIS list since
            # they already exist on other dvlocals or dvnormals
           
            new_list = copy.copy(coef_list)
            for i in xrange(len(coef_list)):

                for idv in xrange(len(self.DV_listNormal)):
                    if coef_list[i] in self.DV_listNormal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
                for idv in xrange(len(self.DV_listLocal)):
                    if coef_list[i] in self.DV_listLocal[idv].coef_list:
                        new_list.remove(coef_list[i])
                    # end if
                # end for
            # end for
            coef_list = new_list
        # end if

        self.DV_listLocal.append(geoDVLocal(\
                dv_name,lower,upper,surf,coef_list,self.g_index))
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1
        
        return


    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,useit=True):
        '''Add a global design variable'''
        self.DV_listGlobal.append(geoDVGlobal(\
                dv_name,value,lower,upper,function,useit))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1
        return 

# ----------------------------------------------------------------------
#                   Surface Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self,file_name,orig=False,surfs=True,coef=True,
                     edges=False,ref_axis=False,links=False,
                     directions=False,labels=False,size=None,nodes=False):

        '''Write the pyGeo Object to Tecplot'''

        # Open File and output header
        print ' '
        print 'Writing Tecplot file: %s '%(file_name)

        f = open(file_name,'w')
        f.write ('VARIABLES = "X", "Y","Z"\n')

        # --------------------------------------
        #    Write out the Interpolated Surfaces
        # --------------------------------------
        
        if surfs == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf].writeTecplotSurface(f,size=size)

        # -------------------------------
        #    Write out the Control Points
        # -------------------------------
        
        if coef == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf].writeTecplotCoef(f)

        # ----------------------------------
        #    Write out the Original Data
        # ----------------------------------
        
        if orig == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf].writeTecplotOrigData(f)
        # ----------------------
        #    Write out the edges
        # ----------------------

        # We also want to output edge continuity for visualization
        if self.con and edges==True:
            counter = 1
            for i in xrange(len(self.con)): #Output Simple Edges (no continuity)
                if self.con[i].cont == 0 and self.con[i].type == 1:
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'simple_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Continuity edges
                if self.con[i].cont == 1 and self.con[i].type == 1:
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'continuity_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Mirror (free) edges
                if self.con[i].type == 0: #output the edge
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'mirror_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for
        # end if

        # ---------------------
        #    Write out Ref Axis
        # ---------------------

        if len(self.ref_axis)>0 and ref_axis==True:
            for r in xrange(len(self.ref_axis)):
                axis_name = 'ref_axis%d'%(r)
                self.ref_axis[r].writeTecplotAxis(f,axis_name)
            # end for
        # end if

        # ------------------
        #    Write out Links
        # ------------------

        if len(self.ref_axis)>0 and links==True:
            for r in xrange(len(self.ref_axis)):
                self.writeTecplotLinks(f,self.ref_axis[r])
            # end for
        # end if
              
        # -----------------------------------
        #    Write out The Surface Directions
        # -----------------------------------

        if directions == True:
            for isurf in xrange(self.nSurf):
                self.surfs[isurf].writeDirections(f,isurf)
            # end for
        # end if

        # ---------------------------------
        #    Write out The Labels
        # ---------------------------------
        if labels == True:
            # Split the filename off
            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'/'+fileBaseName+'.labels.dat'
            f2 = open(label_filename,'w')
            for isurf in xrange(self.nSurf):
                midu = floor(self.surfs[isurf].Nctlu/2)
                midv = floor(self.surfs[isurf].Nctlv/2)
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,ZN=%d, T=\"Surface %d\"\n'%(self.surfs[isurf].coef[midu,midv,0],self.surfs[isurf].coef[midu,midv,1], self.surfs[isurf].coef[midu,midv,2],2*isurf+1,isurf+1)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()

        f.close()
        sys.stdout.write('\n')

        # ---------------------------------
        #    Write out the Node Labels
        # ---------------------------------
        if nodes == True:
            # First we need to figure out where the corners actually *are*
            nodes = zeros((len(self.node_link),3))
            for i in xrange(len(nodes)):
                # Try to find node i
                for isurf in xrange(self.nSurf):
                    if self.node_link[isurf][0] == i:
                        coordinate = self.surfs[isurf].getValueCorner(0)
                        break
                    elif self.node_link[isurf][1] == i:
                        coordinate = self.surfs[isurf].getValueCorner(1)
                        break
                    elif self.node_link[isurf][2] == i:
                        coordinate = self.surfs[isurf].getValueCorner(2)
                        break
                    elif self.node_link[isurf][3] == i:
                        coordinate = self.surfs[isurf].getValueCorner(3)
                        break
                # end for
                nodes[i] = coordinate
            # end for
            # Split the filename off
            (dirName,fileName) = os.path.split(file_name)
            (fileBaseName, fileExtension)=os.path.splitext(fileName)
            label_filename = dirName+'/'+fileBaseName+'.nodes.dat'
            f2 = open(label_filename,'w')

            for i in xrange(len(nodes)):
                text_string = 'TEXT CS=GRID3D, X=%f,Y=%f,Z=%f,T=\"n%d\"\n'%(nodes[i][0],nodes[i][1],nodes[i][2],i)
                f2.write('%s'%(text_string))
            # end for 
            f2.close()

        f.close()
        sys.stdout.write('\n')


        return


    def writeTecplotLinks(self,handle,ref_axis):
        '''Write out the surface links. '''

        num_vectors = len(ref_axis.links_s)
        coords = zeros((2*num_vectors,3))
        icoord = 0
    
        for i in xrange(len(ref_axis.links_s)):
            coords[icoord    ,:] = ref_axis.xs.getValue(ref_axis.links_s[i])
            coords[icoord +1 ,:] = self.coef[ref_axis.coef_list[i]]
            icoord += 2
        # end for

        icoord = 0
        conn = zeros((num_vectors,2))
        for ivector  in xrange(num_vectors):
            conn[ivector,:] = icoord, icoord+1
            icoord += 2
        # end for

        handle.write('Zone N= %d ,E= %d\n'%(2*num_vectors, num_vectors) )
        handle.write('DATAPACKING=BLOCK, ZONETYPE = FELINESEG\n')

        for n in xrange(3):
            for i in  range(2*num_vectors):
                handle.write('%f\n'%(coords[i,n]))
            # end for
        # end for

        for i in range(num_vectors):
            handle.write('%d %d \n'%(conn[i,0]+1,conn[i,1]+1))
        # end for

        return


    def writeIGES(self,file_name):
        '''write the surfaces to IGES format'''
        f = open(file_name,'w')

        #Note: Eventually we may want to put the CORRECT Data here
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,1,4HINCH,8,0.016,15H19970830.165254,0.0001,0.,             G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        
        Dcount = 1;
        Pcount = 1;

        for isurf in xrange(self.nSurf):
            Pcount,Dcount =self.surfs[isurf].writeIGES_directory(\
                f,Dcount,Pcount)

        Pcount  = 1
        counter = 1

        for isurf in xrange(self.nSurf):
            Pcount,counter = self.surfs[isurf].writeIGES_parameters(\
                f,Pcount,counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%7s\n'%(1,4,Dcount-1,counter-1,' ',' '))
        f.close()

        return

    # ----------------------------------------------------------------------
    #                              Utility Functions 
    # ----------------------------------------------------------------------

    def getCoordinatesFromFile(self,file_name):
        '''Get a list of coordinates from a file - useful for testing'''

        f = open(file_name,'r')
        coordinates = []
        for line in f:
            aux = string.split(line)
            coordinates.append([float(aux[0]),float(aux[1]),float(aux[2])])
        # end for
        f.close()
        coordinates = transpose(array(coordinates))

        return coordinates
    
    def attachSurface(self,coordinates,patch_list=None,Nu=20,Nv=20,force_domain=True):

        '''Attach a list of surface points to either all the pyGeo surfaces
        of a subset of the list of surfaces provided by patch_list.

        Arguments:
             coordinates   :  a 3 by nPts numpy array
             patch_list    :  list of patches to locate next to nodes,
                              None means all patches will be used
             Nu,Nv         :  parameters that control the temporary
                              discretization of each surface        
             
        Returns:
             dist          :  distance between mesh location and point
             patchID       :  patch on which each u,v coordinate is defined
             uv            :  u,v coordinates in a 2 by nPts array.

        Modified by GJK to include a search on a subset of surfaces.
        This is useful for associating points in a mesh where points may
        lie on the edges between surfaces. Often, these points must be used
        twice on two different surfaces for load/displacement transfer.        
        '''
        if not self.NO_PRINT:
            print ''
            print 'Attaching a discrete surface to the Geometry Object...'

        if patch_list == None:
            patch_list = range(self.nSurf)
        # end

        nPts = coordinates.shape[1]
        
        # Now make the 'FE' Grid from the sufaces.
        patches = len(patch_list)
        
        nelem    = patches * (Nu-1)*(Nv-1)
        nnode    = patches * Nu *Nv
        conn     = zeros((4,nelem),int)
        xyz      = zeros((3,nnode))
        elemtype = 4*ones(nelem) # All Quads
        
        counter = 0
        for n in xrange(patches):
            isurf = patch_list[n]
            
            u = linspace(self.surfs[isurf].range[0],\
                             self.surfs[isurf].range[1],Nu)
            v = linspace(self.surfs[isurf].range[0],\
                             self.surfs[isurf].range[1],Nv)
            [U,V] = meshgrid(u,v)

            temp = self.surfs[isurf].getValueM(U,V)
            for idim in xrange(self.surfs[isurf].nDim):
                xyz[idim,n*Nu*Nv:(n+1)*Nu*Nv]= \
                    temp[:,:,idim].flatten()
            # end for

            # Now do connectivity info
           
            for j in xrange(Nv-1):
                for i in xrange(Nu-1):
                    conn[0,counter] = Nu*Nv*n + (j  )*Nu + i     + 1
                    conn[1,counter] = Nu*Nv*n + (j  )*Nu + i + 1 + 1 
                    conn[2,counter] = Nu*Nv*n + (j+1)*Nu + i + 1 + 1
                    conn[3,counter] = Nu*Nv*n + (j+1)*Nu + i     + 1
                    counter += 1

                # end for
            # end for
        # end for

        # Now run the csm_pre command 
        if not self.NO_PRINT:
            print 'Running CSM_PRE...'
        [dist,nearest_elem,uvw,base_coord,weightt,weightr] = \
            csm_pre.csm_pre(coordinates,xyz,conn,elemtype)

        # All we need from this is the nearest_elem array and the uvw array

        # First we back out what patch nearest_elem belongs to:
        patchID = (nearest_elem-1) / ((Nu-1)*(Nv-1))  # Integer Division

        # Next we need to figure out what is the actual UV coordinate 
        # on the given surface

        uv = zeros((nPts,2))
        
        for i in xrange(nPts):

            # Local Element
            local_elem = (nearest_elem[i]-1) - patchID[i]*(Nu-1)*(Nv-1)
            #print local_elem
            # Find out what its row/column index is

            #row = int(floor(local_elem / (Nu-1.0)))  # Integer Division
            row = local_elem / (Nu-1)
            col = mod(local_elem,(Nu-1)) 

            #print nearest_elem[i],local_elem,row,col

            u_local = uvw[0,i]
            v_local = uvw[1,i]

            if ( force_domain ):
                if u_local > 1.0:
                    u_local = 1.0
                elif u_local < 0.0:
                    u_local = 0.0
                # end

                if v_local > 1.0:
                    v_local = 1.0
                elif v_local < 0.0:
                    v_local = 0.0
                # end
            # end
            
            uv[i,0] =  u_local/(Nu-1)+ col/(Nu-1.0)
            uv[i,1] =  v_local/(Nv-1)+ row/(Nv-1.0)

        # end for

        # Now go back through and adjust the patchID to the element list
        for i in xrange(nPts):
            patchID[i] = patch_list[patchID[i]]
        # end

        # Release the tree - otherwise fortran will get upset
        csm_pre.release_adt()
        if not self.NO_PRINT:
            print 'Done Surface Attachment'

        return dist,patchID,uv

    def _initdPtdCoef( self, M, Nctl ):
        
        # We know the row filling factor: Its (no more) than ku*kv
        # control points for each control point. Since we don't
        # use more than k=4 we will set at 16
        
        if USE_PETSC:
            dPtdCoef = PETSc.Mat()
            if PETSC_MAJOR_VERSION == 1:
                dPtdCoef.createAIJ([M*3,Nctl*3],nnz=16*3,comm=PETSc.COMM_SELF)
            elif PETSC_MAJOR_VERSION == 0:
                dPtdCoef.createSeqAIJ([M*3,Nctl*3],nz=16*3)
            else:
                print 'Error: PETSC_MAJOR_VERSION = %d is not supported'%(PETSC_MAJOR_VERSION)
                sys.exit(1)
            # end if
        else:
            dPtdCoef = zeros((M*3,Nctl*3))
        # end if

        return dPtdCoef
        

    def calcSurfaceDerivative(self,patchID,uv,indices=None,dPtdCoef=None):
        '''Calculate the (fixed) surface derivative of a discrete set of ponits'''

        if not self.NO_PRINT:
            print 'Calculating Surface Derivative for %d Points...'%(len(patchID))
        timeA = time.time()
        
        if USE_PETSC:
            PETSC_INSERT_MODE = PETSc.InsertMode.ADD_VALUES
        # end if
        # If no matrix is provided, use self.dPtdCoef
        if dPtdCoef == None:
            dPtdCoef = self.dPtdCoef
        # end

        if indices == None:
            indices = arange(len(patchID),'intc')
        # end

        if dPtdCoef == None:
            # Calculate the size Ncoef_free x Ndesign Variables            
            M = len(patchID)
            Nctl = self.Ncoef

            dPtdCoef = self._initdPtdCoef( M, Nctl )
        # end                     
                
        for i in xrange(len(patchID)):
            ku = self.surfs[patchID[i]].ku
            kv = self.surfs[patchID[i]].kv
            Nctlu = self.surfs[patchID[i]].Nctlu
            Nctlv = self.surfs[patchID[i]].Nctlv

            ileftu, mflagu = self.surfs[patchID[i]].pyspline.intrv(\
                self.surfs[patchID[i]].tu,uv[i][0],1)
            ileftv, mflagv = self.surfs[patchID[i]].pyspline.intrv(\
                self.surfs[patchID[i]].tv,uv[i][1],1)

            if mflagu == 0: # Its Inside so everything is ok
                u_list = [ileftu-ku,ileftu-ku+1,ileftu-ku+2,ileftu-ku+3]
            if mflagu == 1: # Its at the right end so just need last one
                u_list = [ileftu-ku-1]

            if mflagv == 0: # Its Inside so everything is ok
                v_list = [ileftv-kv,ileftv-kv+1,ileftv-kv+2,ileftv-kv+3]
            if mflagv == 1: # Its at the right end so just need last one
                v_list = [ileftv-kv-1]

            for ii in xrange(len(u_list)):
                for jj in xrange(len(v_list)):

                    x = self.surfs[patchID[i]].calcPtDeriv(\
                        uv[i][0],uv[i][1],u_list[ii],v_list[jj])

                    index = 3*self.l_index[patchID[i]][u_list[ii],v_list[jj]]
                    if USE_PETSC:
                        dPtdCoef.setValue( 3*indices[i]  , index  ,x,PETSC_INSERT_MODE)
                        dPtdCoef.setValue( 3*indices[i]+1, index+1,x,PETSC_INSERT_MODE)
                        dPtdCoef.setValue( 3*indices[i]+2, index+2,x,PETSC_INSERT_MODE)
                    else:
                        dPtdCoef[3*indices[i]    ,index    ] += x
                        dPtdCoef[3*indices[i] + 1,index + 1] += x
                        dPtdCoef[3*indices[i] + 2,index + 2] += x
                    # end if
                # end for
            # end for
        # end for 
        
        # Assemble the (Constant) dPtdCoef
        if USE_PETSC:
            dPtdCoef.assemblyBegin()
            dPtdCoef.assemblyEnd()
        # end if

        self.dPtdCoef = dPtdCoef # Make sure we're dealing with the same matrix

        if not self.NO_PRINT:
            print 'Finished Surface Derivative in %5.3f seconds'%(time.time()-timeA)

        return

    def createTACSGeo(self,surface_list=None):
        '''
        Create the spline classes for use within TACS
        '''

        try:
            from pyTACS import elements as elems
        except:
            print 'Could not import TACS. Cannot create TACS splines.'
            return
        # end

        if USE_PETSC == False:
            print 'Must have PETSc to create TACS splines.'
            return
        # end

        if surface_list == None:
            surface_list = arange(self.nSurf)
        # end if

        # Calculate the Number of global design variables
        N = 0
        for i in xrange(len(self.DV_listGlobal)): #Global Variables
            if self.DV_listGlobal[i].useit:
                N += self.DV_listGlobal[i].nVal
            # end if
        # end for

        gdvs = arange(N,dtype='intc')
      
        global_geo = elems.GlobalGeo( gdvs, self.petsc_coef, self.dCoefdx )
      
        # For each dv object, number the normal variables
        normalDVs = []
        for normal in self.DV_listNormal:
            normalDVs.append( arange(N,N+normal.nVal,dtype='intc') )
            N += normal.nVal
        # end

        # For each dv object, number all three coordinates
        localDVs = []
        for local in self.DV_listLocal:
            localDVs.append( arange(N,N+3*local.nVal,dtype='intc') )
            N += 3*local.nVal
        # end

        # Create the list of local dvs for each surface patch
        surfDVs = []
        for i in xrange(self.nSurf):
            surfDVs.append(None)
        # end
        
        for i in xrange(len(self.DV_listNormal)):
            sid = self.DV_listNormal[i].surface_id
            if ( surfDVs[sid] == None ):
                surfDVs[sid] = normalDVs[i]
            else:
                hstack( surfDVs[sid], normalDVs[i] )
            # end
        # end

        for i in xrange(len(self.DV_listLocal)):
            sid = self.DV_listLocal[i].surface_id
            if ( surfDVs[sid] == None ):
                surfDVs[sid] = localDVs[i]
            else:
                hstack( surfDVs[sid], localDVs[i] )
            # end
        # end        

        # Go through and add local objects for each design variable
        def convert( isurf, ldvs ):
            if ldvs == None:
                ldvs = []
            # end

            return elems.SplineGeo( int(self.surfs[isurf].ku),
                                    int(self.surfs[isurf].kv),
                                    self.surfs[isurf].tu, self.surfs[isurf].tv,
                                    self.surfs[isurf].coef[:,:,0], 
                                    self.surfs[isurf].coef[:,:,1], 
                                    self.surfs[isurf].coef[:,:,2], 
                                    global_geo, ldvs, self.l_index[isurf].astype('intc') )
        # end

        tacs_surfs = []
        for isurf in surface_list:
            tacs_surfs.append( convert(isurf, surfDVs[isurf] ) )
        # end
     
        return global_geo, tacs_surfs


class edge(object):
    '''A class for edge objects'''

    def __init__(self,n1,n2,cont,degen,intersect,dg,Nctl):
        self.n1        = n1        # Integer for node 1
        self.n2        = n2        # Integer for node 2
        self.cont      = cont      # Integer: 0 for c0 continuity, 1 for c1 continuity
        self.degen     = degen     # Integer: 1 for degenerate, 0 otherwise
        self.intersect = intersect # Integer: 1 for an intersected edge, 0 otherwise
        self.dg        = dg        # Design Group index
        self.Nctl      = Nctl      # Number of control points for this edge

    def write_info(self,i,handle):
        handle.write('  %3d          |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |  %3d |\n'\
                         %(i,self.n1,self.n2,self.cont,self.degen,self.intersect,self.dg,self.Nctl))
                                                                          
class ref_axis(object):

    def __init__(self,X,rot,*args,**kwargs):

        ''' Create a generic reference axis. This object bascally defines a
        set of points in space (x,y,z) each with three rotations
        associated with it. The purpose of the ref_axis is to link
        groups of b-spline controls points together such that
        high-level planform-type variables can be used as design
        variables
        
        Input:

        X: array of size N,3: Contains the x-y-z coodinates of the axis
        rot: array of size N,3: Contains the rotations of the axis

        Note: Rotations are performed in the order: Z-Y-X
        '''

        self.links_s = []
        self.links_x = []
        self.con_type = None
        if not  X.shape == rot.shape:
            print 'The shape of X and rot must be the same'
            print 'X:',X.shape
            print 'rot:',rot.shape
            sys.exit(1)

        # Note: Ref_axis data is ALWAYS Complex. 
        X = X.astype('D')
        rot = rot.astype('D')
        self.N = X.shape[0]

        self.base_point = X[0,:]
        
        self.base_point_s = None
        self.base_point_D = None

        self.end_point   = X[-1,:]
        self.end_point_s = None
        self.end_point_D = None

        # Values are stored wrt the base point
        self.x = X-self.base_point
        self.rot = rot
        self.scale = ones(self.N,'D')

        # Deep copy the x,rot and scale for design variable reference
        self.x0 = copy.deepcopy(self.x)
        self.rot0 = copy.deepcopy(self.rot)
        self.scale0 = copy.deepcopy(self.scale)

        # Create an interpolating spline for the spatial part and use
        # its  basis for the rotatinoal part
        
        self.xs = pySpline.linear_spline(\
            task='interpolate',X=self.base_point+self.x,k=2,complex=True)
        self.s = self.xs.s

        self.rotxs = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,0],k=2,s=self.s,complex=True)
        self.rotys = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,1],k=2,s=self.s,complex=True)
        self.rotzs = pySpline.linear_spline(\
            task='interpolate',X=self.rot[:,2],k=2,s=self.s,complex=True)

        self.scales = pySpline.linear_spline(\
            task='interpolate',X=self.scale,k=2,s=self.s,complex=True)

    def update(self):
        
        self.xs.coef = self.base_point+self.x
        self.rotxs.coef = self.rot[:,0]
        self.rotys.coef = self.rot[:,1]
        self.rotzs.coef = self.rot[:,2]

        self.scales.coef = self.scale

        if self.con_type == 'full':
            self.xs.coef[-1,:] = self.end_point
        # end if
        
        return
       
    def writeTecplotAxis(self,handle,axis_name):
        '''Write the ref axis to the open file handle'''
        N = len(self.s)
        handle.write('Zone T=%s I=%d\n'%(axis_name,N))
        values = self.xs.getValueV(self.s)
        for i in xrange(N):
            handle.write('%f %f %f \n'%(values[i,0],values[i,1],values[i,2]))
        # end for

        return

    def getRotMatrixGlobalToLocal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return     dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s))))
    
    def getRotMatrixLocalToGlobal(self,s):
        
        '''Return the rotation matrix to convert vector from global to
        local frames'''
        return inv(dot(rotyM(self.rotys(s)),dot(rotxM(self.rotxs(s)),\
                                                    rotzM(self.rotzs(s)))))
    
class geoDVGlobal(object):
     
    def __init__(self,dv_name,value,lower,upper,function,useit):
        
        '''Create a geometric design variable (or design variable group)

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        value: Value of Design Variable
        
        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable. '''

        self.name = dv_name
        self.value = value
        if isinstance(value, int):
            self.nVal = 1
        else:
            self.nVal = len(value)

        self.lower    = lower
        self.upper    = upper
        self.function = function
        self.useit    = useit
        return

    def __call__(self,ref_axis):

        '''When the object is called, actually apply the function'''
        # Run the user-supplied function
        return self.function(self.value,ref_axis)
        

class geoDVNormal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,coef_list,g_index,l_index):
        
        '''Create a set of gemoetric design variables which change the shape
        of surface, surface_id

        Input:
        
        dv_name: Design variable name. Must be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        surface_id: The surface these design variables apply to 

        coef_list: The list of (global) indicies for thes design variables

        global_coef: The g_index list from pyGeo

        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''

        self.nVal = len(coef_list)
        self.value = zeros(self.nVal,'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        self.coef_list = coef_list
        self.l_index   = l_index[surface_id]
        # We also need to know what local surface i,j index is for
        # each point in the coef_list since we need to know the
        # position on the surface to get the normal. That's why we
        # passed in the global_coef list so we can figure it out
        
        self.local_coef_index = zeros((self.nVal,2),'intc')
        
        for icoef in xrange(self.nVal):
            current_point = g_index[coef_list[icoef]]
            # Since the local DV only have driving control points, the
            # i,j index coorsponding to the first entryin the
            # global_coef list is the one we want
            self.local_coef_index[icoef,:] = g_index[coef_list[icoef]][0][1:3]
        # end for
        return

    def __call__(self,surface,coef):

        '''When the object is called, apply the design variable values to the
        surface'''

        coef = pySpline.pyspline_cs.updatesurfacepoints(\
            coef,self.local_coef_index,self.coef_list,self.value,\
                self.l_index,surface.tu,surface.tv,surface.ku,surface.kv)

        return coef

    def getNormals(self,surf,coef):
        normals = pySpline.pyspline_real.getctlnormals(\
            coef,self.local_coef_index,self.coef_list,\
                self.l_indexs,surf.tu,surf.tv,surf.ku,surf.kv)
        return normals

    def removeCoef(self,rm_list):
        '''Remove coefficient from this dv if its in rm_list'''
        for i in xrange(len(rm_list)):
            if rm_list[i] in self.coef_list:
                index = self.coef_list.index(rm_list[i])
                del self.coef_list[index]
                delete(self.local_coef_index,index)
                delete(self.value,index)
                self.nVal -= 1
            # end if
        # end for

        return

class geoDVLocal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id,coef_list,global_coef):
        
        '''Create a set of gemoetric design variables whcih change the shape
        of a surface surface_id. Local design variables change the surface
        in all three axis.

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        surface_id: Surface this set of design variables belongs to

        coef_list: The indicies on the surface used for these dvs

        global_coef: The pyGeo global_design variable linkinng list to
        determine if a design variable is free of driven
        
        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''

        self.nVal = len(coef_list)
        self.value = zeros((3*self.nVal),'D')
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        self.coef_list = coef_list
        
        # We also need to know what local surface i,j index is for
        # each point in the coef_list since we need to know the
        # position on the surface to get the normal. That's why we
        # passed in the global_coef list so we can figure it out
        
        self.local_coef_index = zeros((self.nVal,2),'intc')
        
        for icoef in xrange(self.nVal):
            self.local_coef_index[icoef,:] = global_coef[coef_list[icoef]][0][1:3]
        # end for
        return

    def __call__(self,coef):

        '''When the object is called, apply the design variable values to 
        coefficients'''
        
        for i in xrange(self.nVal):
            coef[self.coef_list[i]] += self.value[3*i:3*i+3]
        # end for
      
        return coef

    def removeCoef(self,rm_list):
        '''Remove coefficient from this dv if its in rm_list'''
        for i in xrange(len(rm_list)):
            if rm_list[i] in self.coef_list:
                index = self.coef_list.index(rm_list[i])
                del self.coef_list[index]
                delete(self.local_coef_index,index)
                delete(self.value,index)
                self.nVal -= 1
            # end if
        # end for
   
class point_select(object):

    def __init__(self,type,*args,**kwargs):

        '''Initialize a control point selection class. There are several ways
        to initialize this class depending on the 'type' qualifier:

        Inputs:
        
        type: string which inidicates the initialization type:
        
        'x': Define two corners (pt1=,pt2=) on a plane parallel to the
        x=0 plane

        'y': Define two corners (pt1=,pt2=) on a plane parallel to the
        y=0 plane

        'z': Define two corners (pt1=,pt2=) on a plane parallel to the
        z=0 plane

        'quad': Define FOUR corners (pt1=,pt2=,pt3=,pt4=) in a
        COUNTER-CLOCKWISE orientation 

        'slice': Define a grided region using two slice parameters:
        slice_u= and slice_v are used as inputs

        'list': Simply use a list of control point indidicies to
        use. Use coef = [[i1,j1],[i2,j2],[i3,j3]] format

        '''
        
        if type == 'x' or type == 'y' or type == 'z':
            assert 'pt1' in kwargs and 'pt2' in kwargs,'Error:, two points \
must be specified with initialization type x,y, or z. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2]'

        elif type == 'quad':
            assert 'pt1' in kwargs and 'pt2' in kwargs and 'pt3' in kwargs \
                and 'pt4' in kwargs,'Error:, four points \
must be specified with initialization type quad. Points are specified \
with kwargs pt1=[x1,y1,z1],pt2=[x2,y2,z2],pt3=[x3,y3,z3],pt4=[x4,y4,z4]'
            
        elif type == 'slice':
            assert 'slice_u'  in kwargs and 'slice_v' in kwargs,'Error: two \
python slice objects must be specified with slice_u=slice1, slice_v=slice_2 \
for slice type initialization'

        elif type == 'list':
            assert 'coef' in kwargs,'Error: a coefficient list must be \
speficied in the following format: coef = [[i1,j1],[i2,j2],[i3,j3]]'
        else:
            print 'Error: type must be one of: x,y,z,quad,slice or list'
            sys.exit(1)
        # end if

        if type == 'x' or type == 'y' or type =='z' or type == 'quad':
            corners = zeros([4,3])
            if type == 'x':
                corners[0] = kwargs['pt1']

                corners[1][1] = kwargs['pt2'][1]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][1] = kwargs['pt1'][1]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:,0] = 0.5*(kwargs['pt1'][0] + kwargs['pt2'][0])

            elif type == 'y':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][2] = kwargs['pt1'][2]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][2] = kwargs['pt2'][2]

                corners[3] = kwargs['pt2']

                corners[:,1] = 0.5*(kwargs['pt1'][1] + kwargs['pt2'][1])

            elif type == 'z':
                corners[0] = kwargs['pt1']

                corners[1][0] = kwargs['pt2'][0]
                corners[1][1] = kwargs['pt1'][1]

                corners[2][0] = kwargs['pt1'][0]
                corners[2][1] = kwargs['pt2'][1]

                corners[3] = kwargs['pt2']

                corners[:,2] = 0.5*(kwargs['pt1'][2] + kwargs['pt2'][2])

            elif type == 'quad':
                corners[0] = kwargs['pt1']
                corners[1] = kwargs['pt2']
                corners[2] = kwargs['pt4'] # Note the switch here from CC orientation
                corners[3] = kwargs['pt3']
            # end if

            X = reshape(corners,[2,2,3])

            self.box=pySpline.surf_spline(task='lms',ku=2,kv=2,\
                                              Nctlu=2,Nctlv=2,X=X)

        elif type == 'slice':
            self.slice_u = kwargs['slice_u']
            self.slice_v = kwargs['slice_v']
        elif type == 'list':
            self.coef_list = kwargs['coef']
        # end if

        self.type = type

        return


    def getControlPoints(self,surface,surface_id,coef_list,l_index):

        '''Take is a pySpline surface, and a (possibly non-empty) coef_list
        and add to the coef_list the global index of the control point
        on the surface that can be projected onto the box'''
        
        if self.type=='x'or self.type=='y' or self.type=='z' or self.type=='quad':

            for i in xrange(surface.Nctlu):
                for j in xrange(surface.Nctlv):
                    u0,v0,D,converged = self.box.projectPoint(surface.coef[i,j])
                    if u0 > 0 and u0 < 1 and v0 > 0 and v0 < 1: # Its Inside
                        coef_list.append(l_index[surface_id][i,j])
                    #end if
                # end for
            # end for
        elif self.type == 'slice':
            for i in self.slice_u:
                for j in self.slice_v:
                    coef_list.append(l_index[surface_id][i,j])
                # end for
            # end for
        elif self.type == 'list':
            for i in xrange(len(self.coef_list)):
                coef_list.append(l_index[surface_id][self.coef_list[i][0],
                                                     self.coef_list[i][1]])
            # end for
        # end if

        return coef_list

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'

