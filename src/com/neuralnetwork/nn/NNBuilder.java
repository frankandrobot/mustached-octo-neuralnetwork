package com.neuralnetwork.nn;

import com.neuralnetwork.core.interfaces.INeuralLayer;

import java.util.ArrayList;
import java.util.Arrays;

public class NNBuilder
{
    INeuralLayer[] aLayers;

    public NNBuilder setLayers(INeuralLayer... aLayers)
    {
        ArrayList<INeuralLayer> layers = new ArrayList<INeuralLayer>();

        if (this.aLayers != null && this.aLayers.length > 0)
            layers.addAll(Arrays.asList(this.aLayers));

        layers.addAll(Arrays.asList(aLayers));

        this.aLayers = layers.toArray(new INeuralLayer[layers.size()]);

        return this;
    }

    public INeuralLayer[] getLayers()
    {
        return aLayers;
    }

    public NN build() {

        return new NN(this);
    }
}