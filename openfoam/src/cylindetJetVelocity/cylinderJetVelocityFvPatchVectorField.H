/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::cylinderJetVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a rotational velocity condition.
    The boundary condition was originally implemented by Tom Krogmann
    during his student project at TU Braunschweig. For more information
    refer to:
        GitHub: https://github.com/TomKrogmann/Optimal_Sensor_Placement_for_Active_Flow_Control_in_Deep_Reinforcement_Learning
        report: https://zenodo.org/record/7636959#.Y-qTLcbMJkg

Usage
    \table
        Property     | Description             | Required    | Default value
        origin       | origin of rotation in Cartesian coordinates | yes|
        axis         | axis of rotation        | yes         |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            cylinderJetVelocity;
        origin_a        (1 0 0);
        origin_b        (0 1 0);
        origin_c        (0 -1 0);
        axis            (0 0 1);
        policy          "policy.pt"
        seed            0;
        train           true;
        absOmegaMax     5.0;
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    cylinderJetVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef cylinderJetVelocityFvPatchVectorField_H
#define cylinderJetVelocityFvPatchVectorField_H

#include <torch/script.h>
#include <random>
#include "fixedValueFvPatchFields.H"
#include "probes.H"
#include "timeControl.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class cylinderJetVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class cylinderJetVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- origin of cylinder
        vector origin_;

        //- training or evaluation mode
        bool train_;

        //- name of the PyTorch angular velocity model
        word policy_name_;

        //- PyTorch model predicting mean and log(std) of angular velocity
        torch::jit::script::Module policy_;

        //- largest allowed absolute value of angular velocity
        scalar abs_Q_max_;

        //- seed for random sampling
        int seed_;

        //- name of the probes function object dictionary
        word probes_name_;

        //- random number generator
        std::mt19937 gen_;

        //- next predicted angular velocity of cylinder a
        scalar Q_upper_;

        //- previously predicted angular velocity of cylinder a
        scalar Q_old_upper_;

        //- next predicted angular velocity of cylinder b
        scalar Q_lower_;

        //- previously predicted angular velocity of cylinder b
        scalar Q_old_lower_;

        //- last time at which angular velocity was computed
        scalar control_time_;

        //- update of next angular velocity value if true
        bool update_Q_;

        //- when to start controlling
        scalar start_time_;

        //- time increment between control actions
        scalar dt_control_;

        //- probes function object
        Foam::probes probes_;

        //- timeControls consistent with function objects
        Foam::timeControl control_;
        
        //- lists of face centers, normals, and face IDs
        DynamicList <vector> centers_upper_;
        DynamicList <vector> centers_lower_;
        DynamicList <vector> normals_upper_;
        DynamicList <vector> normals_lower_;
        DynamicList <label> faces_upper_;
        DynamicList <label> faces_lower_;

public:

    //- Runtime type information
    TypeName("cylinderJetVelocity");


    // Constructors

        //- Construct from patch and internal field
        cylinderJetVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        cylinderJetVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given cylinderJetVelocityFvPatchVectorField
        //  onto a new patch
        cylinderJetVelocityFvPatchVectorField
        (
            const cylinderJetVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        cylinderJetVelocityFvPatchVectorField
        (
            const cylinderJetVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new cylinderJetVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        cylinderJetVelocityFvPatchVectorField
        (
            const cylinderJetVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
             const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new cylinderJetVelocityFvPatchVectorField(*this, iF)
            );
        }



    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;

        //- Save trajectory to file
        void saveTrajectory(scalar alpha_upper, scalar beta_upper, scalar alpha_lower, scalar beta_lower) const;

        //- Get probes function object dictionary
        const Foam::dictionary& getProbesDict();

        //- initialize probes function object
        Foam::probes initializeProbes();

        //- initialize control
        Foam::timeControl initializeControl();

        //- determine attribution of patch faces to cylinders a/b/c
        void initializeJetSegmentation();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
