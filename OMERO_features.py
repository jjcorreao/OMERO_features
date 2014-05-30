__author__ = 'jcorrea'

# TODO: new version version in progress for farmer
# weka_segmentation.py
# weka segmentation general

# for SGNWORKER
IMAGEJ_PATH="/global/project/projectdirs/ngbi/resources/ImageJ/ImageJ-linux64"
XVFBRUN_PATH="/global/project/projectdirs/ngbi/resources/xvfb-run"
OMERO_PATH="/global/project/projectdirs/ngbi/omero-0.7/OMERO.server/bin/omero"
PBS_GEN="/global/project/projectdirs/ngbi/jobs/jobgen-dev.sh"
# MACRO_PATH = "/global/project/projectdirs/ngbi/jobs/ij_macros/trainable_weka.ijm"
MACRO_PATH = "/global/project/projectdirs/ngbi/jobs/ij_macros/stack_out.ijm"
MACRO_PATH2 = "/global/project/projectdirs/ngbi/jobs/ij_macros/trainable_weka_farmer.ijm"
OMERO_FS="/project/projectdirs/ngbi/omero5/OMERO.data/ManagedRepository"

# MACRO_PATH = "/global/project/projectdirs/ngbi/jobs/ij_macros/trainable_weka_segmentation.ijm"


GSCRATCH="/global/scratch2/sd/jcorrea"
cache_dir = "/global/scratch2/sd/jcorrea/ngbi/tmp"

import omero
import pickle
import os
from omero.gateway import BlitzGateway
import omero.scripts as scripts
from omero.rtypes import *
import omero.util.script_utils as script_utils
import numpy as np
from numpy import zeros, hstack, vstack
import sys, traceback, subprocess, os

import time
import timeit

import matplotlib as mpl
import scipy as sp
import PIL as pil
import numpy as np

import omero.cli as cli

from omero.rtypes import rtime, rlong, rstring, rlist, rint
from omero_model_ExperimenterI import ExperimenterI
from omero_model_ExperimenterGroupI import ExperimenterGroupI
from omero_model_PermissionsI import PermissionsI

import tempfile
import subprocess

import math

def features(conn, scriptParams, uuid):
    pass

    cli = omero.cli.CLI()
    cli.loadplugins()

    feature_type = scriptParams["feature_type"]
    CP_pipe_params = scriptParams["CP_params"]

    images, logMessage = script_utils.getObjects(conn, scriptParams)
    if not images:
        return None, None, logMessage

    imageIds = [i.getId() for i in images]

    for iId in imageIds:
        image = conn.getObject("Image", iId)
        dataset = image.getParent().getId()
        image_dir = os.path.join(OMERO_FS, image.getFileset().listFiles()[0].getPath()) # Given a single file per fileset
        image_name = image.getFileset().listFiles()[0].getName()
        image_path = os.path.join(image_dir, image_name)
        
        # CellProfiler context manager

        # CP_PATH='/project/projectdirs/ngbi/resources/CP2'
        # my_image_directory='/project/projectdirs/ngbi/jobs/CP_pipes/test/test_in'
        # my_output_directory='/project/projectdirs/ngbi/jobs/CP_pipes/test/test_out'
        # # my_pipe='/project/projectdirs/ngbi/jobs/CP_pipes/pipeline.h5'
        # my_pipe='/project/projectdirs/ngbi/jobs/CP_pipes/pipeline.cppipe'
        CP_args = (CP_PATH, image_path, obj_mask_path, CP_pipe)
        CP_process_str = "python %s/CellProfiler.py -c -r -i %s -o obj_mask_path -p %s" % CP_args # using sgnworker
        
        new_dId, new_dObj = dataset_gen(conn, "datasetname_name")

        # read obj_mask_path and split into objects
        for image_object in object_list:
#           crop image
            cmd = "import -s sgn02 -d %s -n %s -k %s %s" % (new_dId, image_object.getName(), key, image)
            u=cmd.split()
            cli.invoke(u,strict=True)
            new_iId, new_iObj = image_gen(conn, image_object.getName())
            link2project(conn, new_dObj, new_iObj)
    
    #           save image and assotiate with iId
    #           save object as image using omeroFS
    #           create annotation link etc
    #           calculate feature of object
    #           picke feature and store
                

        new_pId, new_pObj = project_gen(conn, image_object.getName())
        
        link2project(conn, new_pObj, new_dObj)
        addTag(conn, new_dId, tags, chans)
        # job_liner=[]

        # all_jobs = open("%s.job" % (os.path.join(cache_dir, tmpdir_out.split('/')[-1])), 'w+')


    # for image in images OK
    #   for objects in image OK
    #       for object in objects
    #           save object as image using omeroFS
    #           create annotation link etc
    #           calculate feature of object
    #           picke feature and store


    # from objects in image
        # crop objects
        # save 

def addTag(conn, datasetId, tags, chans):

    updateService = conn.getUpdateService()

    datasetObj = conn.getObject("Dataset", datasetId)

    imageObjs = list(datasetObj.listChildren())
    imageIds = [imageObjs[i].getId() for i in range(len(imageObjs))]

    for imageId in imageIds:

        image = conn.getObject("Image", imageId)._obj

        for tag in tags:
            t = conn.getObjects("TagAnnotation", attributes={'textValue':tag})
            t = list(t)

            if len(t) != 0:

                annotation = t[0]._obj

                image = image.__class__(image.id.val, False)
                l = omero.model.ImageAnnotationLinkI()
                l.setParent(image)
                l.setChild(annotation)
                updateService.saveAndReturnObject(l)

            else:

                annotation = omero.model.TagAnnotationI()
                annotation.setTextValue(rstring(tag))
                image = image.__class__(image.id.val, False)
                l = omero.model.ImageAnnotationLinkI()
                l.setParent(image)
                l.setChild(annotation)
                updateService.saveAndReturnObject(l)

def image_gen(conn, name):
    imageObj = omero.model.ImageI()
    imageObj.setName(rstring(name))
    imageObj = conn.getUpdateService().saveAndReturnObject(imageObj)
    imageId = imageObj.getId().getValue()
    # print "New image, Id:", imageId
    return imageId, imageObj

def dataset_gen(conn, name):
    datasetObj = omero.model.DatasetI()
    datasetObj.setName(rstring(name))
    datasetObj = conn.getUpdateService().saveAndReturnObject(datasetObj)
    datasetId = datasetObj.getId().getValue()
    # print "New dataset, Id:", datasetId
    return datasetId, datasetObj

def project_gen(conn, name):
    projectObj = omero.model.ProjectI()
    projectObj.setName(rstring(name))
    projectObj = conn.getUpdateService().saveAndReturnObject(projectObj)
    projectId = projectObj.getId().getValue()
    # print "New project, Id:", projectId
    return projectId, projectObj

def link2project(conn, projectObj, datasetObj):

    if projectObj is None:
        sys.stderr.write("Error: Object does not exist.\n")
        sys.exit(1)
    link = omero.model.ProjectDatasetLinkI()
    link.setParent(omero.model.ProjectI(projectObj.getId(), False))
    link.setChild(datasetObj)
    conn.getUpdateService().saveObject(link)

def write_features_to_file():
    pass

def vocabulary():
    pass    

def weka_segmentation(conn, scriptParams, uuid):


    model = scriptParams["Segmentation_model"]
    model_path=model

    user = conn.getUser()
    user = user.getName()
    print("user: %s" % (user))

    print(model_path)

    images, logMessage = script_utils.getObjects(conn, scriptParams)
    if not images:
        return None, None, logMessage
    imageIds = [i.getId() for i in images]

    for iId in imageIds:

        tmpdir_stack = tempfile.mkdtemp(dir=cache_dir)
        tmpdir_out = tempfile.mkdtemp(dir=cache_dir)

        image = conn.getObject("Image", iId)
        dataset = image.getParent().getId()

        sizeZ = image.getSizeZ()
        print(sizeZ)

        # job_liner=[]

        # all_jobs = open("%s.job" % (os.path.join(cache_dir, tmpdir_out.split('/')[-1])), 'w+')

        image_dir = os.path.join(OMERO_FS, image.getFileset().listFiles()[0].getPath())
        image_name = image.getFileset().listFiles()[0].getName()
        image_path = os.path.join(image_dir, image_name)

        ijmacro_args = "%s:%s/:%s" % (image_path, tmpdir_out, model_path)
        
        memory_ij=int(math.ceil((2.00*sizeZ)))

        all_jobs = open("%s.job" % (os.path.join(cache_dir, tmpdir_out.split('/')[-1])), 'w+')
        job_liner="%s -a %s -Xmx%sg -- -macro %s %s -batch" % (XVFBRUN_PATH, IMAGEJ_PATH, memory_ij, MACRO_PATH2, ijmacro_args)
        all_jobs.writelines(job_liner)

        print(job_liner)

        system = scriptParams["System"]
        wtime = scriptParams["Wall_time"]
        pmem = scriptParams["Private_memory"]
        bigmem = scriptParams["Big_memory_nodes"]
        nodes = scriptParams["Nodes"]
        ppn = scriptParams["PPN"]

        if bigmem:
            print("big memory nodes flag")
        else:

        # ijmacro_args = "%s/:%s/:%s" % (tmpdir_stack, tmpdir_out, model_path)
        pbs_file = "%s.pbs" % (os.path.join(cache_dir, tmpdir_out.split('/')[-1]))

        nodes = int(math.ceil(((2.00*sizeZ)+0.15*(2.00*sizeZ))/48))

        stack_args = "%s/" % (tmpdir_out)
        qsub_cmd = ". %s %s %s %s %s %s %s %s %s %s %s %s > %s" % (PBS_GEN, user, dataset, image_name, uuid, MACRO_PATH, stack_args, tmpdir_out, wtime, pmem, all_jobs.name, nodes, pbs_file)
        # $ ./jobgen.sh <user> <dataset> <name> <uuid> <ijmacro> <ijmacro args> <outpath> <wtime> <pmem> <all_jobs> <no_nodes> > job.pbs

        print(qsub_cmd)
        os.system(qsub_cmd)

        enableKeepAlive_time = (36*60*60)
        conn.c.enableKeepAlive(enableKeepAlive_time)
        os.system("ssh %s '/usr/syscom/opt/torque/4.2.6/bin/qsub %s'" % (system, pbs_file))

def pbsgen(ij_args):
    print(hello)


def runAsScript():

    dataTypes = [rstring('Image')]

    models_path="/project/projectdirs/ngbi/jobs/ij_macros/classifiers"
    systems=['carver', 'sgnworker']

    # modelPath = []
    segmentationModel = []
    for file in os.listdir(models_path):
        if file.endswith(".model"):
            segmentationModel.append(str(os.path.join(models_path,file)))
            # segmentationModel.append(file)

    # Include files metadata information; link to a web application

    # segmentationModel = ['Desulfovibrio RCH1', 'Geobacter', 'Myxococcus Xanthus1', 'Myxococcus Xanthus2']

    client = scripts.client('weka_serial.py', """Segment a dataset using Random Forest and a known classifier""",

    scripts.String("Data_Type", optional=False, grouping="1",
        description="Pick Images by 'Image' ID or by the ID of their 'Dataset'", values=dataTypes, default="Image"),

    scripts.List("IDs", optional=False, grouping="1.1",
        description="List of Dataset IDs or Image IDs to process.").ofType(rlong(0)),

    scripts.String("Segmentation_model", optional=False, grouping="2",
        description="Select model", values=segmentationModel, default=segmentationModel[0]),

    scripts.String("System", optional=False, grouping="3",
        description="Select the system", values=systems, default=systems[0]),

    scripts.String("Wall_time", grouping="3.1",
        description="Wall time", default='0:30:00'),

    scripts.String("Private_memory", grouping="3.2",
        description="Private memory", default='4GB'),

    scripts.Bool("Big_memory_nodes", grouping="3.2.1",
        description="Big memory nodes", default='False'),

    scripts.String("Nodes", grouping="3.3",
        description="Nodes", default='1'),

    scripts.String("PPN", grouping="3.4",
        description="PPN", default='5'),

    version = "0",
    authors = ["Joaquin Correa", "OSP Team"],
    institutions = ["NERSC"],
    contact = "JoaquinCorrea@lbl.gov",
    )

    try:
        session = client.getSession();





        # print("session key: %s" % (session)

        # process the list of args above.
        scriptParams = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                scriptParams[key] = client.getInput(key, unwrap=True)

        print scriptParams

        print("key: %s" % (key))
        # print("session key: %s" % (session))

        # wrap client to use the Blitz Gateway
        conn = BlitzGateway(client_obj=client)

        admin = conn.getAdminService()
        uuid = admin.getEventContext().sessionUuid
        # uuid = uuid.split(":")[1]


        print(uuid)

        # print("key: %s" % (key))

        # sizeZ, sizeC, sizeT, sizeX, sizeY, dataset = newFilteredImage(conn, scriptParams)
        # newFilteredImage(conn, scriptParams)

        # sloopy("/global/project/projectdirs/ngbi/omero-0.7/OMERO.server/lib/scripts/omero/tmp/WT_7B_duct2.mrc")
        weka_segmentation(conn, scriptParams, uuid)

        # print sizeZ, sizeC, sizeT, sizeX, sizeY, dataset

        # Return message, new image and new dataset (if applicable) to the client
        # client.setOutput("Message", rstring(message))
        # if len(images) == 1:
        #     client.setOutput("Image", robject(images[0]._obj))
        # if dataset is not None:
        #     client.setOutput("New Dataset", robject(dataset._obj))

    finally:
        client.closeSession()

if __name__ == "__main__":
    runAsScript()

