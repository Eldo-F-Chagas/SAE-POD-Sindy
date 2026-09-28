/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
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

\*---------------------------------------------------------------------------*/

#include "codedFunctionObjectTemplate.H"
#include "volFields.H"
#include "read.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(errorFunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    errorFunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 1115dcd2f4aa5080265d6fa260b04b958a922563
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void error_1115dcd2f4aa5080265d6fa260b04b958a922563(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const fvMesh& errorFunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

errorFunctionObject::errorFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::regionFunctionObject(name, runTime, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

errorFunctionObject::~errorFunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool errorFunctionObject::read(const dictionary& dict)
{
    if (false)
    {
        Info<<"read error sha1: 1115dcd2f4aa5080265d6fa260b04b958a922563\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


Foam::wordList errorFunctionObject::fields() const
{
    if (false)
    {
        Info<<"fields error sha1: 1115dcd2f4aa5080265d6fa260b04b958a922563\n";
    }

    wordList fields;
//{{{ begin code
    
//}}} end code

    return fields;
}


bool errorFunctionObject::execute()
{
    if (false)
    {
        Info<<"execute error sha1: 1115dcd2f4aa5080265d6fa260b04b958a922563\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool errorFunctionObject::write()
{
    if (false)
    {
        Info<<"write error sha1: 1115dcd2f4aa5080265d6fa260b04b958a922563\n";
    }

//{{{ begin code
    
//}}} end code

    return true;
}


bool errorFunctionObject::end()
{
    if (false)
    {
        Info<<"end error sha1: 1115dcd2f4aa5080265d6fa260b04b958a922563\n";
    }

//{{{ begin code
    #line 111 "/home/mlep-pc-ubuntu/OpenFOAM/mlep-pc-ubuntu-13/run/cylinder/system/functions/error"

        // Lookup U
        Info<< "Looking up field U\n" << endl;
        const volVectorField& U = mesh().lookupObject<volVectorField>("U");

        Info<< "Reading inlet velocity uInfX\n" << endl;

        scalar ULeft = 0.0;
        label leftI = mesh().boundaryMesh().findIndex("left");
        const fvPatchVectorField& fvp = U.boundaryField()[leftI];
        if (fvp.size())
        {
            ULeft = fvp[0].x();
        }
        reduce(ULeft, maxOp<scalar>());

        dimensionedScalar uInfX
        (
            "uInfx",
            dimensionSet(0, 1, -1, 0, 0),
            ULeft
        );

        Info << "U at inlet = " << uInfX.value() << " m/s" << endl;

        scalar magCylinder = 0.0;
        label cylI = mesh().boundaryMesh().findIndex("cylinder");
        const fvPatchVectorField& cylFvp = mesh().C().boundaryField()[cylI];
        if (cylFvp.size())
        {
            magCylinder = mag(cylFvp[0]);
        }
        reduce(magCylinder, maxOp<scalar>());

        dimensionedScalar radius
        (
            "radius",
            dimensionSet(0, 1, 0, 0, 0),
            magCylinder
        );

        Info << "Cylinder radius = " << radius.value() << " m" << endl;

        volVectorField UA
        (
            IOobject
            (
                "UA",
                mesh().time().name(),
                U.mesh(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            U
        );

        Info<< "\nEvaluating analytical solution" << endl;

        const volVectorField& centres = UA.mesh().C();
        volScalarField magCentres(mag(centres));
        volScalarField theta(acos((centres & vector(1,0,0))/magCentres));

        volVectorField cs2theta
        (
            cos(2*theta)*vector(1,0,0)
          + sin(2*theta)*vector(0,1,0)
        );

        UA = uInfX*(dimensionedVector(vector(1,0,0))
          - pow((radius/magCentres),2)*cs2theta);

        // Force writing of UA (since time has not changed)
        UA.write();

        volScalarField error("error", mag(U-UA)/mag(UA));

        Info<<"Writing relative error in U to " << error.objectPath()
            << endl;

        error.write();
    
//}}} end code

    return true;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

