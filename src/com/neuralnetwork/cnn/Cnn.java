package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.INeuralLayer;

public class Cnn<T>
{
    private final INeuralLayer<T>[] aLayers;

    public Cnn(CnnBuilder netBuilder)
    {
        this.aLayers = netBuilder.getLayers();

        //check layers output---inputs must match
        for(int i=0; i<aLayers.length-1; i++)
        {
            INeuralLayer layer = aLayers[i];
            INeuralLayer layerNext = aLayers[i+1];

            if (layer.getOutputDim() != layerNext.getInputDim())
                throw new IllegalArgumentException("Output/inputs in layers must match: layers "+i+","+i+1);
        }
    }

    public T generateOutput(T input)
    {
        T _input = input;
        T _output = null;

        for(int i=0; i<aLayers.length; i++)
        {
            _output = aLayers[i].generateOutput(_input);
            _input = _output;
        }
        return _output;
    }
}
