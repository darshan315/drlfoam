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

\*---------------------------------------------------------------------------*/
#include "cylinderJetVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::cylinderJetVelocityFvPatchVectorField::
    cylinderJetVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(p, iF),
      origin_(),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeJetSegmentation();
}

Foam::cylinderJetVelocityFvPatchVectorField::
    cylinderJetVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const dictionary &dict)
    : fixedValueFvPatchField<vector>(p, iF, dict, false),
      origin_(dict.get<vector>("origin_")),
      train_(dict.get<bool>("train")),
      policy_name_(dict.get<word>("policy")),
      policy_(torch::jit::load(policy_name_)),
      abs_Q_max_(dict.get<scalar>("absQMax")),
      seed_(dict.get<int>("seed")),
      probes_name_(dict.get<word>("probesDict")),
      gen_(seed_),
      Q_upper_(0.0),
      Q_old_upper_(0),
      Q_lower_(0.0),
      Q_old_lower_(0),
      control_time_(0.0),
      update_Q_(true),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeJetSegmentation();
    updateCoeffs();
}


Foam::cylinderJetVelocityFvPatchVectorField::
    cylinderJetVelocityFvPatchVectorField(
        const cylinderJetVelocityFvPatchVectorField &ptf,
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const fvPatchFieldMapper &mapper)
    : fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
      origin_(ptf.origin_),
      train_(ptf.train_),
      policy_name_(ptf.policy_name_),
      policy_(ptf.policy_),
      abs_Q_max_(ptf.abs_Q_max_),
      seed_(ptf.seed_),
      probes_name_(ptf.probes_name_),
      gen_(ptf.gen_),
      Q_upper_(ptf.Q_upper_),
      Q_old_upper_(ptf.Q_old_upper_),
      Q_lower_(ptf.Q_lower_),
      Q_old_lower_(ptf.Q_old_lower_),
      control_time_(ptf.control_time_),
      update_Q_(ptf.update_Q_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeJetSegmentation();
}

Foam::cylinderJetVelocityFvPatchVectorField::
    cylinderJetVelocityFvPatchVectorField(
        const cylinderJetVelocityFvPatchVectorField &rwvpvf)
    : fixedValueFvPatchField<vector>(rwvpvf),
      origin_(rwvpvf.origin_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_Q_max_(rwvpvf.abs_Q_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      Q_upper_(rwvpvf.Q_upper_),
      Q_old_upper_(rwvpvf.Q_old_upper_),
      Q_lower_(rwvpvf.Q_lower_),
      Q_old_lower_(rwvpvf.Q_old_lower_),
      control_time_(rwvpvf.control_time_),
      update_Q_(rwvpvf.update_Q_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeJetSegmentation();
}

Foam::cylinderJetVelocityFvPatchVectorField::
    cylinderJetVelocityFvPatchVectorField(
        const cylinderJetVelocityFvPatchVectorField &rwvpvf,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(rwvpvf, iF),
      origin_(rwvpvf.origin_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_Q_max_(rwvpvf.abs_Q_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      Q_upper_(rwvpvf.Q_upper_),
      Q_old_upper_(rwvpvf.Q_old_upper_),
      Q_lower_(rwvpvf.Q_lower_),
      Q_old_lower_(rwvpvf.Q_old_lower_),
      control_time_(rwvpvf.control_time_),
      update_Q_(rwvpvf.update_Q_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeJetSegmentation();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::cylinderJetVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }
    
    // update angular velocities
    const scalar t = this->db().time().timeOutputValue();
    const scalar dt = this->db().time().deltaTValue();
    const scalar timeIndex = this->db().time().timeIndex();
    const scalar startTimeIndex = this->db().time().startTimeIndex();
    bool timeToControl = control_.execute() &&
                         t >= start_time_ - 0.5*dt &&
                         timeIndex != startTimeIndex;
                         
    if (timeToControl && update_Q_)
    {
    	Q_old_upper_ = Q_upper_;
        Q_old_lower_ = Q_lower_;
        control_time_ = t;
        
        const volScalarField& p = this->db().lookupObject<volScalarField>("p"); 
        scalarField p_sample = probes_.sample(p);

	if (Pstream::master()) // evaluate policy only on the master
        {
            torch::Tensor features = torch::from_blob(
                p_sample.data(), {1, p_sample.size()}, torch::TensorOptions().dtype(torch::kFloat64)
            );
            
            std::vector<torch::jit::IValue> policyFeatures{features};
            torch::Tensor dist_parameters = policy_.forward(policyFeatures).toTensor();
            scalar alpha_upper = dist_parameters[0][0].item<double>();
            scalar alpha_lower = dist_parameters[0][1].item<double>();
            scalar beta_upper = dist_parameters[0][3].item<double>();
            scalar beta_lower = dist_parameters[0][4].item<double>();
            std::gamma_distribution<double> distribution_1_a(alpha_upper, 1.0);
            std::gamma_distribution<double> distribution_2_a(alpha_lower, 1.0);
            std::gamma_distribution<double> distribution_1_b(beta_upper, 1.0);
            std::gamma_distribution<double> distribution_2_b(beta_lower, 1.0);
            scalar Q_pre_scale_upper;
            scalar Q_pre_scale_lower;
            
            if (train_)
            {
                // sample from Beta distribution during training
                double number_1_a = distribution_1_a(gen_);
                double number_2_a = distribution_2_a(gen_);
                double number_1_b = distribution_1_b(gen_);
                double number_2_b = distribution_2_b(gen_);
                Q_pre_scale_upper = number_1_a / (number_1_a + number_1_b);
                Q_pre_scale_lower = number_2_a / (number_2_a + number_2_b);
            }
            else
            {
                // use expected (mean) angular velocity
                Q_pre_scale_upper = alpha_upper / (alpha_upper + beta_upper);
                Q_pre_scale_lower = alpha_lower / (alpha_lower + beta_lower);
            }
            // rescale to actionspace
            Q_upper_ = (Q_pre_scale_upper - 0.5) * 2 * abs_Q_max_;
            Q_lower_ = (Q_pre_scale_lower - 0.5) * 2 * abs_Q_max_;
            // save trajectory
            saveTrajectory(alpha_upper, beta_upper, alpha_lower, beta_lower);
            Info << "New Q_upper: " << Q_upper_ << "; old value: " << Q_old_upper_ << "\n";
            Info << "New Q_lower: " << Q_lower_ << "; old value: " << Q_old_lower_ << "\n";
        }
        
        Pstream::scatter(Q_upper_);
        Pstream::scatter(Q_lower_);
        // avoid update of angular velocity during p-U coupling
        update_Q_ = false;
    }
    
    // activate update of angular velocity after p-U coupling
    if (!timeToControl && !update_Q_)
    {
        update_Q_ = true;
    }

    
    // update angular velocity by linear transition from old to new value
    scalar d_Q_upper = (Q_upper_ - Q_old_upper_) / dt_control_ * (t - control_time_);
    scalar Q_upper = Q_old_upper_ + d_Q_upper;
    scalar d_Q_lower = (Q_lower_ - Q_old_lower_) / dt_control_ * (t - control_time_);
    scalar Q_lower = Q_old_lower_ + d_Q_lower;
    
    const int patch_size = (patch().Cf()).size();
    vectorField Up(patch_size);
    
    	//Applying velocity to the faces
    	// 1. get velocity from Q -> Umax
	// 		Umax : Q/area => Q/(thickness * 2*R*sin(jet_angle))
    	// 2. parabolic velocity profile for jet
    	//		vel_y = Umax * (1 - ((x-c)^2  / 2r^2)) 
    	// 3. get c and r
    	//		r : length of jet patch => 2*R_cylinder*sin(jet_angle)
    	//		c : center of jet => centre of cylinder
    	// 4. vector(0,vec_y,0)
    
    scalar thickness_ = 0.01;
    scalar c_x = 0.2;
    scalar jet_length = 0.00523359562429438327221186296090; //2*R_cylinder*sin(jet_angle);
    scalar jet_area = thickness_ * jet_length;
    //scalar Q_upper = 0.0003;
        			
    forAll(faces_upper_, faceI)
    {
    	label faceUI = faces_upper_[faceI];
	scalar U_max_up = Q_upper / jet_area;
	scalar u_parabl_y_up = U_max_up*(1-((pow((centers_upper_[faceI].x() - c_x)/jet_length,2))/2));
	Up[faceUI] = vector(0,u_parabl_y_up,0);
    	
    }

    forAll(faces_lower_, faceI)
    
    {
	label faceLI = faces_lower_[faceI];
	scalar U_max_lw = Q_lower / jet_area;
	scalar u_parabl_y_lw = U_max_lw*(1-((pow((centers_lower_[faceI].x() - c_x)/jet_length,2))/2));
	Up[faceLI] = vector(0,-1*u_parabl_y_lw,0);
    	
    }

    vectorField::operator=(Up);
    fixedValueFvPatchVectorField::updateCoeffs();
    
}


void Foam::cylinderJetVelocityFvPatchVectorField::write(Ostream &os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin_", origin_);
    os.writeEntry("policy", policy_name_);
    os.writeEntry("train", train_);
    os.writeEntry("absQMax", abs_Q_max_);
    os.writeEntry("seed", seed_);
    os.writeEntry("probesDict", probes_name_);
}

void Foam::cylinderJetVelocityFvPatchVectorField::saveTrajectory(scalar alpha_upper, scalar beta_upper, scalar alpha_lower, scalar beta_lower) const
{
    std::ifstream file("trajectory.csv");
    std::fstream trajectory("trajectory.csv", std::ios::app | std::ios::binary);
    const scalar t = this->db().time().timeOutputValue();

    if(!file.good())
    {
        // write header
        trajectory << "t, Q_upper, alpha_upper, beta_upper, Q_lower, alpha_lower, beta_lower";
    }

    trajectory << std::setprecision(15)
               << "\n"
               << t << ", "
               << Q_upper_ << ", "
               << alpha_upper << ", "
               << beta_upper << ", "
               << Q_lower_ << ", "
               << alpha_lower << ", "
               << beta_lower;
               

}

const Foam::dictionary& Foam::cylinderJetVelocityFvPatchVectorField::getProbesDict()
{
    const dictionary& funcDict = this->db().time().controlDict().subDict("functions");
    if (!funcDict.found(probes_name_))
    {
        FatalError << "probesDict" << probes_name_ << " not found\n" << exit(FatalError);
        
    }
    return funcDict.subDict(probes_name_);
}

Foam::probes Foam::cylinderJetVelocityFvPatchVectorField::initializeProbes()
{
    const dictionary& probesDict = getProbesDict();
    return Foam::probes("probes", this->db().time(), probesDict, false, true);
}

Foam::timeControl Foam::cylinderJetVelocityFvPatchVectorField::initializeControl()
{
    const dictionary& probesDict = getProbesDict();
    start_time_ = probesDict.getOrDefault<scalar>("timeStart", 0.0);
    dt_control_ = probesDict.getOrDefault<scalar>("executeInterval", 1.0);
    return Foam::timeControl(this->db().time(), probesDict, "execute");
}

void Foam::cylinderJetVelocityFvPatchVectorField::initializeJetSegmentation()
{
    const fvMesh& mesh(patch().boundaryMesh().mesh());
    // patch name and radius are currently hardcoded
    label patchID = mesh.boundaryMesh().findPatchID("cylinders");
    //scalar radius = 0.5;
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    const surfaceScalarField& magSf = mesh.magSf();
    const surfaceVectorField& Cf = mesh.Cf();
    const surfaceVectorField& Sf = mesh.Sf();

    forAll(cPatch, faceI)
    {
        scalar y = Cf.boundaryField()[patchID][faceI].y();
    	scalar o_y_ = origin_[1];

        if (y < o_y_) //lower
        {
            centers_upper_.append(Cf.boundaryField()[patchID][faceI]);
            normals_upper_.append(Sf.boundaryField()[patchID][faceI]/magSf.boundaryField()[patchID][faceI]);
            faces_upper_.append(faceI);
        }

        if (y > o_y_) //upper
        {
            centers_lower_.append(Cf.boundaryField()[patchID][faceI]);
            normals_lower_.append(Sf.boundaryField()[patchID][faceI]/magSf.boundaryField()[patchID][faceI]);
            faces_lower_.append(faceI);
        }

    }

}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField(
        fvPatchVectorField,
        cylinderJetVelocityFvPatchVectorField);
}

// ************************************************************************* //
