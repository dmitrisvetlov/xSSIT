function [sens,constraintBounds, stateSpace] = computeSensitivity(model,...
    parameters,...
    tout,...
    fspTol,...
    initialStates,...
    initialProbabilities,...
    constraintFunctions, constraintBounds,...
    isTimeInvariant, verbose, useMex, method, app, stateSpace)
arguments
    model
    parameters
    tout
    fspTol
    initialStates
    initialProbabilities
    constraintFunctions
    constraintBounds
    isTimeInvariant
    verbose
    useMex
    method
    app
    stateSpace =[];
end
        
app.SensParDropDown.Items = model.parameterNames;

if (strcmp(method, 'forward'))
    %     try
    if ~isempty(parameters)
        parsDict = containers.Map(parameters(:,1),...
            parameters(:,2));
    else
        parsDict=[];
    end
    propensities = model.createPropensities(parsDict,app.ReactionsTabOutputs.varNames);
    [propensityDerivatives,computableSensitivities] = model.findPropensityDerivativesSymbolic(parsDict,app.ReactionsTabOutputs.varNames);

    app.SensParDropDown.Items = model.parameterNames(find(computableSensitivities));

    stoichMatrix = model.stoichiometry;

    relTol=1.0e-10 ;
    absTol=1.0e-6 ;
    initialSensitivities = zeros(size(initialStates,2)*sum(computableSensitivities), 1);
    [Outputs,constraintBounds,stateSpace] = ssit.sensitivity.adaptiveFspForwardSens(tout, initialStates,...
        initialProbabilities, initialSensitivities,...
        stoichMatrix,...
        propensities, propensityDerivatives, computableSensitivities,...
        fspTol,...
        constraintFunctions, constraintBounds,...
        verbose, useMex, relTol, absTol, stateSpace);

    sens = struct(...
        'format', 'forward',...
        'data', {Outputs}...
        );
    return
    %     catch
    %         disp('Error with Analytical Sensitivity Calculations - Switching to Finite Difference Method')
    %         app.FiniteDifferenceButton.Value = 1;
    %         app.SensitivityFunctionButton.Value = 0;
    %     end
end

% If the forward sensitivity did not work, try finite difference.
app.SensParDropDown.Items = model.parameterNames;
perturbationFactor = 1.0e-4;

[outputs,constraintBounds] = ssit.sensitivity.adaptiveFspFiniteDiff(...
    model,...
    parameters,...
    perturbationFactor,...
    tout,...
    initialStates,...
    initialProbabilities,...
    fspTol,...
    constraintFunctions, constraintBounds,...
    verbose);

sens = struct(...
    'format', 'finitediff',...
    'data', {outputs}...
    );
end