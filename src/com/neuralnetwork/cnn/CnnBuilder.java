package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.core.interfaces.INeuralNetwork;

public class CnnBuilder<T>
{
    private INeuralLayer<T>[] aLayers;

    /*private double learningParam;
    private double momentumParam;*/

    public CnnBuilder setLayers(INeuralLayer... aLayers)
    {
        this.aLayers = aLayers;
        return this;
    }

    public INeuralLayer[] getLayers()
    {
        return aLayers;
    }

/*
    public CnnBuilder setLearningParam(double learningParam)
    {
        this.learningParam = learningParam;
        return this;
    }

    public double getLearningParam()
    {
        return learningParam;
    }

    public CnnBuilder setMomentumParam(double momentumParam)
    {
        this.momentumParam = momentumParam;
        return this;
    }

    public double getMomentumParam()
    {
        return momentumParam;
    }
*/

    public Cnn build() {

        return new Cnn(this);
    }
}
