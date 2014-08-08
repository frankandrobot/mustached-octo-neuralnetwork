package com.neuralnetwork.nn;

import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.core.interfaces.INeuralNetwork;

public class MultiLayerNN implements INeuralNetwork<double[]>
{
    private final INeuralLayer[] aLayers;

    public MultiLayerNN(MultiLayerNNBuilder netBuilder)
    {
        this.aLayers = netBuilder.getLayers();

        //check layers output---inputs must match
        for(int i=0; i<aLayers.length-1; i++)
        {
            INeuralLayer layer = aLayers[i];
            INeuralLayer layerNext = aLayers[i+1];

            if (layer.getOutputDim() + 1 != layerNext.getInputDim())
                throw new IllegalArgumentException("Output/inputs in layers must match: layers "+i+","+(i+1));
        }
    }

    public double[] generateOutput(double[] input)
    {
        double[] _input = input;
        double[] _output = null;

        for(int i=0; i<aLayers.length; i++)
        {
            _output = aLayers[i].generateY(_input);
            _input = _output;
        }
        return _output;
    }
}
