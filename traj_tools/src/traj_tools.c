/**
 * Langevin dynamics in 1D with PMF and viscosity.
 * 
 * Copyright 2016 Christopher T. Lee <ctlee@ucsd.edu>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */ 

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include "gasdev.h"
#include "pcg_variants.h"
#include "entropy.h"

double kb = 0.0013806488; // kgA^2s^-2K^-1

/* Dummy function pointer holders */
static PyObject* pmf = NULL;
static PyObject* nu = NULL;

/**
 * A test of the PCG random number generator.
 * @param nums: nparray to be filled
 * @param N: len(nums)
 * @returns: int success
 */
static PyObject* testRNG(PyObject* self, PyObject* args){
    PyArrayObject *nums;
    int N;
    if (!PyArg_ParseTuple(args, "O!i", 
                &PyArray_Type, &nums,      
                &N))
        return NULL;
    // Initialize and seed the PCG RNG
    pcg64_random_t rng;
    pcg128_t seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg64_srandom_r(&rng, seeds[0], seeds[1]);
    // Generate N numbers
    for (int i = 0; i < N; i++){
        // Gaussian distributed
        *((double *) PyArray_GETPTR1(nums,i)) =  gasdev(&rng);
        // Uniform floating point
        //*((double *) PyArray_GETPTR1(nums,i)) = random_real(&rng);
    }
    return Py_BuildValue("i", 0);
}

static PyObject* testForce(PyObject* self, PyObject* args)
{   
    PyObject *pmf_cb, *arglist, *result;
    double force;
    if (PyArg_ParseTuple(args, "O", &pmf_cb)) {
        
        if (!PyCallable_Check(pmf_cb)) {
            PyErr_SetString(PyExc_TypeError, "pmf must be callable");
            return NULL;
        } 
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "unable to parse args");
        return NULL;
    }
   
    Py_XINCREF(pmf_cb); // Add a reference to the new callback
    Py_XDECREF(pmf);    // dispose of the previous callback
    pmf = pmf_cb;       // remember the new callback
    
    for (double i = -25; i <= 25; i += 0.1){
        // set the initial force 
        arglist = Py_BuildValue("(d)", i); 
        result =  PyObject_CallObject(pmf, arglist);
        
        Py_DECREF(arglist);
        force = PyFloat_AsDouble(result); 
        Py_DECREF(result);
        printf("Force at %f: %f\n", i, force);
    }
    return Py_BuildValue("i", 0);
}

/**
 * Heavily optimized langevin integrator. Some optimizations assume
 * correct Python callback response. This may make the code unstable
 * if something goes wrong in Python.
 * @param pmf_cb: PMF callback
 * @param nu_cb: Viscosity callback
 * @param m: mass
 * @param r: hydrodynamic radius
 * @param T: temperature
 * @param pos: starting position
 * @param vel: starting velocity
 * @param min: min milestone
 * @param max: max milestone
 * @param dt: timestep
 * @param N: maxsteps to run
 * @return: bool accept, double final position, double time
 */
static PyObject* milestoneO(PyObject* self, PyObject* args)
{   
    PyObject *pmf_cb, *nu_cb, *arglist;
    double m, r, T, pos, vel, min, max, dt, force, viscosity;
    int reverse, reflecting;
    long int N;
    if (PyArg_ParseTuple(args, "OOddddddddlii",
            &pmf_cb, &nu_cb, &m, &r, &T, &pos,  
            &vel, &min, &max, &dt, &N, &reverse, &reflecting)) {
        
        if (!PyCallable_Check(pmf_cb)) {
            PyErr_SetString(PyExc_TypeError, "pmf must be callable");
            return NULL;
        } 
        if (!PyCallable_Check(nu_cb)) {
            PyErr_SetString(PyExc_TypeError, "nu must be callable");
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "unable to parse args");
        return NULL;
    }
   
    //printf("mass %e; radius %f; temp %f; pos %f; ", m, r, T, pos);
    //printf("vel %e; min %f; max %f; dt %e; N %d; reverse %d; "
    //        , vel, min, max, dt, N, reverse);
    //printf("reflecting %d\n", reflecting);
   
    // Handle the callback function pointers
    Py_XINCREF(pmf_cb); // Add a reference to the new callback
    Py_XDECREF(pmf);    // dispose of the previous callback
    pmf = pmf_cb;       // remember the new callback
    
    Py_XINCREF(nu_cb);
    Py_XDECREF(nu);
    nu = nu_cb;

    PyObject *result;
    double a, b, bdt, bdtmm, c, cdtmm, noise, fnew, bump;
    double mm = 2*m;    // precalculate 2m
    double sixpir = 6*M_PI*r;
    double xkbTdt = 2*kb*T*dt;
    double startpos = pos;
    double oldpos = pos;
    
    // set the initial force 
    arglist = Py_BuildValue("(d)", pos); 
    result =  PyObject_CallObject(pmf, arglist);
    Py_DECREF(arglist);
    force = PyFloat_AsDouble(result); 
    Py_DECREF(result);
    
    pcg64_random_t rng;
    pcg128_t seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg64_srandom_r(&rng, seeds[0], seeds[1]);
    
    // Begin iteration
    for (long int i = 1; i <= N; i++) {
        // Get the local viscosity and calculate constants
        arglist = Py_BuildValue("(d)", pos);
        result = PyObject_CallObject(nu, arglist);
        Py_DECREF(arglist);
        viscosity = *(double *) PyArray_GETPTR1((PyArrayObject*) result, 0); 
        Py_DECREF(result);

        c = sixpir*viscosity;
        cdtmm = (c*dt)/mm; 
        b = 1/(1+cdtmm);
        a = (1-cdtmm)/(1+cdtmm);
        bdt = b*dt;
        bdtmm = bdt/mm;
        
        noise = sqrt(c*xkbTdt);  // Variance of the gaussian noise
        bump = noise * gasdev(&rng);
        /**
         * This integrator is an implementation of:
         * Gronbech-Jensen, N., Farago, O., 
         * Molecular Physics, V111, 8:989-991, (2013)
         */
        pos = pos + bdt*vel + bdtmm*dt*force + bdtmm*bump;
        
        // Update force
        arglist = Py_BuildValue("(d)", pos); 
        result = PyObject_CallObject(pmf, arglist);
        Py_DECREF(arglist);
        fnew = PyFloat_AsDouble(result); 
        Py_DECREF(result);
      
        vel = a*vel + dt/mm*(a*force + fnew) +  b/m*bump;
        force = fnew;

        //printf("%ld %e %e\n", i, pos, vel);
        if (reverse && i > 1) {
            // if self crossing then reject
            if ((pos - startpos > 0) != (oldpos - startpos > 0)) {
                return Py_BuildValue("ifl", 0, pos, i);
            }
        }
        if (pos < min || pos > max){
            if (reflecting && pos < min){
                pos = min + (min - pos);
                vel = -vel;
            }
            else if (pos < min)
                return Py_BuildValue("ifl", 1, min, i);
            else if (pos > max)
                return Py_BuildValue("ifl", 1, max, i);
        }
        oldpos = pos;
    }
    return Py_BuildValue("ifl", 0, pos, -1);
}

/**
 * Run a sample trajectory with some milestone criteria.
 * @param pmf_cb: PMF callback
 * @param nu_cb: Viscosity callback
 * @param m: mass
 * @param r: hydrodynamic radius
 * @param T: temperature
 * @param pos: starting position
 * @param vel: starting velocity
 * @param min: min milestone
 * @param max: max milestone
 * @param dt: timestep
 * @param N: maxsteps to run
 * @return: bool accept, double final position, double time
 */
static PyObject* milestone(PyObject* self, PyObject* args)
{   
    PyObject *pmf_cb, *nu_cb, *arglist;
    double m, r, T, pos, vel, min, max, dt, force, viscosity;
    int reverse, reflecting;
    long int N;
    if (PyArg_ParseTuple(args, "OOddddddddlii",
            &pmf_cb, &nu_cb, &m, &r, &T, &pos,  
            &vel, &min, &max, &dt, &N, &reverse, &reflecting)) {
        
        if (!PyCallable_Check(pmf_cb)) {
            PyErr_SetString(PyExc_TypeError, "pmf must be callable");
            return NULL;
        } 
        if (!PyCallable_Check(nu_cb)) {
            PyErr_SetString(PyExc_TypeError, "nu must be callable");
            return NULL;
        }
    }
    else {
        return NULL;
    }
    //printf("mass %e; radius %f; temp %f; pos %f; vel %e; min %f; max %f; dt %e; N %d; reverse %d; reflecting %d\n", m, r, T, pos, vel, min, max, dt, N, reverse, reflecting);
   
    // Handle the callback function pointers
    Py_XINCREF(pmf_cb); // Add a reference to the new callback
    Py_XDECREF(pmf);    // dispose of the previous callback
    pmf = pmf_cb;       // remember the new callback
    
    Py_XINCREF(nu_cb);
    Py_XDECREF(nu);
    nu = nu_cb;

    PyObject *result;
    double a, b, bdt, bdtmm, c, cdtmm, noise, fnew;
    double mm = 2*m;    // precalculate 2m
    double sixpir = 6*M_PI*r;
    double startpos = pos;
    double oldpos = pos;
    
    // set the initial force 
    arglist = Py_BuildValue("(d)", pos); 
    result =  PyObject_CallObject(pmf, arglist);
    Py_DECREF(arglist);

    if (result && PyFloat_Check(result)) {
        force = PyFloat_AsDouble(result); 
        //printf("Initial force: %e kgAs^-2\n", force); 
    }               
    else {
        printf("PMF.force() returned: %s\n", PyString_AsString(PyObject_Repr(result)));
        PyErr_SetString(PyExc_TypeError, "PMF.force() has returned a non-float");
        return NULL;
    }
    Py_DECREF(result);

    double bump;

    pcg64_random_t rng;
    pcg128_t seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg64_srandom_r(&rng, seeds[0], seeds[1]);
    
    // Begin iteration
    for (long int i = 1; i <= N; i++) {

        // Get the local viscosity and calculate constants
        arglist = Py_BuildValue("(d)", pos);
        result = PyObject_CallObject(nu, arglist);
        Py_DECREF(arglist);
        
        if (result && PyArray_Check(result)) {
            viscosity = *(double *) PyArray_GETPTR1((PyArrayObject*) result, 0); 
            //printf("Viscosity value: %e kgA^-1s^-1\n", viscosity);
        }
        else {
            printf("Viscosity() returned: %s\n", PyString_AsString(PyObject_Repr(result)));
            PyErr_SetString(PyExc_TypeError, "Viscosity has returned a non-array");
            return NULL;
        }
        Py_DECREF(result);

        c = sixpir*viscosity;
        cdtmm = (c*dt)/mm; 
        b = 1/(1+cdtmm);
        a = (1-cdtmm)/(1+cdtmm);
        bdt = b*dt;
        bdtmm = bdt/mm;
    
        noise = sqrt(2*c*kb*T*dt);  // Variance of the gaussian noise
        double gaussnoise = gasdev(&rng);
        printf("%e\n", gaussnoise);
        //bump = noise * gasdev(&rng);
        bump = noise * gaussnoise;
        /**
         *This integrator is an implementation of:
         *Gronbech-Jensen, N., Farago, O., Molecular Physics, V111, 8:989-991, (2013)
         */
        pos = pos + bdt*vel + bdtmm*dt*force + bdtmm*bump;
        
        // Update force
        arglist = Py_BuildValue("(d)", pos); 
        result = PyObject_CallObject(pmf, arglist);
        Py_DECREF(arglist);

        if (result && PyFloat_Check(result)) {
            fnew = PyFloat_AsDouble(result); 
            //printf("New force: %e kgAs^-2\n", fnew); 
        }
        else {
            printf("PMF.force() returned: %s\n", PyString_AsString(PyObject_Repr(result)));
            PyErr_SetString(PyExc_TypeError, "PMF.force() has returned a non-float");
            return NULL;
        }
        Py_DECREF(result);
        
        vel = a*vel + dt/mm*(a*force + fnew) +  b/m*bump;
        force = fnew;

        //printf("step: %d position: %e vel: %e\n", i, pos, vel);
        if (reverse && i > 1) {
            // if self crossing then reject
            if ((pos - startpos > 0) != (oldpos - startpos > 0)) {
                return Py_BuildValue("ifl", 0, pos, i);
            }
        }
        if (pos < min || pos > max){
            if (reflecting && pos < min){
                pos = min + (min - pos);
                vel = -vel;
            }
            else if (pos < min)
                return Py_BuildValue("ifl", 1, min, i);
            else if (pos > max)
                return Py_BuildValue("ifl", 1, max, i);
        }
        oldpos = pos;
    }
    return Py_BuildValue("ifl", 0, pos, -1);
}

/*  define functions in module */
static PyMethodDef TrajTools[] =
{
    {"testRNG", testRNG, METH_VARARGS, "Generate random numbers"},
    {"testForce", testForce, METH_VARARGS, "Test force across profile"},
    {"milestoneO", milestoneO, METH_VARARGS, "Run optimized milestoning"},
    {"milestone", milestone, METH_VARARGS, "Run milestoning"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
inittraj_tools(void)
{
    (void) Py_InitModule("traj_tools", TrajTools);
    import_array();
}
