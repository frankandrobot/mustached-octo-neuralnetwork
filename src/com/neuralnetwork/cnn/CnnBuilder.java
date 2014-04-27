package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;

public class CnnBuilder
{
    private IActivationFunction globalActivationFunction;
    private INeuralNetwork.IMatrixNeuralNetwork[] aLayers;
    private double learningParam;
    private double momentumParam;

    public CnnBuilder setGlobalActivationFunction(IActivationFunction globalActivationFunction)
    {
        this.globalActivationFunction = globalActivationFunction;
        return this;
    }

    public IActivationFunction getGlobalActivationFunction()
    {
        return globalActivationFunction;
    }

    public CnnBuilder setLayers(INeuralNetwork.IMatrixNeuralNetwork... aLayers)
    {
        this.aLayers = aLayers;
        return this;
    }

    public INeuralNetwork.IMatrixNeuralNetwork[] getLayers()
    {
        return aLayers;
    }

    public CnnBuilder setLearningParam(double learningParam) {
        this.learningParam = learningParam;
        return this;
    }

    public double getLearningParam()
    {
        return learningParam;
    }

    public CnnBuilder setMomentumParam(double momentumParam) {
        this.momentumParam = momentumParam;
        return this;
    }

    public double getMomentumParam() {
        return momentumParam;
    }
}
