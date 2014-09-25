package com.neuralnetwork.nn;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import com.neuralnetwork.core.interfaces.INnLayer;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NN implements INeuralNetwork<double[],double[]>
{
    private final INnLayer[] aLayers;

    public NN(NNBuilder netBuilder)
    {
        this.aLayers = netBuilder.getLayers();

        //check layers output---inputs must match
        for(int i=0; i<aLayers.length-1; i++)
        {
            INnLayer layer = aLayers[i];
            INnLayer layerNext = aLayers[i+1];

            assertThat("layer input/output must match in layer "+i+","+(i+1),
                    layer.getOutputDim().rows + 1,
                    is(layerNext.getInputDim().rows));
        }
    }

    public INnLayer[] getLayers()
    {
        return aLayers;
    }

    public double[] generateYoutput(double... input)
    {
        double[] _input = input;
        double[] y = null;

        for(int i=0; i<aLayers.length; i++)
        {
            y = aLayers[i].generateY(_input);
            _input = y;
        }

        return y;
    }

    public double[] generateOutput(double... input)
    {
        double[] y = generateYoutput(input);

        double[] output = new double[y.length-1];
        System.arraycopy(y,1,output,0,output.length);

        return output;

    }
}
