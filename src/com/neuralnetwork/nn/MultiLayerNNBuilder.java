package com.neuralnetwork.nn;

import com.neuralnetwork.core.interfaces.INeuralLayer;

public class MultiLayerNNBuilder
{
    INeuralLayer[] aLayers;

    public MultiLayerNNBuilder setLayers(INeuralLayer... aLayers)
    {
        this.aLayers = aLayers;
        return this;
    }

    public INeuralLayer[] getLayers()
    {
        return aLayers;
    }

    public MultiLayerNN build() {

        return new MultiLayerNN(this);
    }
}
