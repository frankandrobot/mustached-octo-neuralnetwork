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

    public INeuralLayer[] getLayers()
    {
        return aLayers;
    }

    public double[] generateOutput(double[] input)
    {
        double[] _input = input;
        double[] y = null;

        for(int i=0; i<aLayers.length; i++)
        {
            y = aLayers[i].generateY(_input);
            _input = y;
        }

        double[] output = new double[y.length-1];
        System.arraycopy(y,1,output,0,output.length);

        return output;
    }
}
