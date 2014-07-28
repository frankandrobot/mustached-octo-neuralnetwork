package com.neuralnetwork.core;

import com.neuralnetwork.core.interfaces.INeuralLayer;

public class MultiLayerNetworkBuilder<T>
{
    private INeuralLayer<T>[] aLayers;

    /*private double learningParam;
    private double momentumParam;*/

    public MultiLayerNetworkBuilder setLayers(INeuralLayer... aLayers)
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

    public MultiLayerNetwork build() {

        return new MultiLayerNetwork(this);
    }
}
