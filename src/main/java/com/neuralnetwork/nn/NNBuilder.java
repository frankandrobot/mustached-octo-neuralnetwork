package com.neuralnetwork.nn;

import com.neuralnetwork.core.interfaces.INnLayer;

import java.util.ArrayList;
import java.util.Arrays;

public class NNBuilder
{
    INnLayer[] aLayers;

    public NNBuilder setLayers(INnLayer... aLayers)
    {
        ArrayList<INnLayer> layers = new ArrayList<INnLayer>();

        if (this.aLayers != null && this.aLayers.length > 0)
            layers.addAll(Arrays.asList(this.aLayers));

        layers.addAll(Arrays.asList(aLayers));

        this.aLayers = layers.toArray(new INnLayer[layers.size()]);

        return this;
    }

    public INnLayer[] getLayers()
    {
        return aLayers;
    }

    public NN build() {

        return new NN(this);
    }
}
