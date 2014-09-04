package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.filter.SimpleConvolutionFilter;
import com.neuralnetwork.cnn.layer.builder.ConvolutionLayerBuilder;
import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.neuron.MNeuron;
import org.ejml.data.DenseMatrix64F;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class ConvolutionLayerTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    private DenseMatrix64F input;
    private double[] weights;
    private ConvolutionLayerBuilder builder;

    @Before
    public void setup()
    {
        input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        weights = new double[]{0.1, 0.2, 0.3, 0.4, 0.5};

        builder = new ConvolutionLayerBuilder()
            .setNeuron(new MNeuron(phi, weights))
            .set1DInputSize(3)
            .setFilter(new SimpleConvolutionFilter());
     }

    @Test
    public void testConvolution1()
    {
        ConvolutionLayer layer = builder.build();

        DenseMatrix64F output = layer.generateOutput(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(output.numRows, is(2) );
        assertThat(output.numCols, is(2) );
    }

    @Test
    public void testConvolution2()
    {
        ConvolutionLayer layer = builder.build();

        double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];
        o11 = phi.apply(o11);

        double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];
        o12 = phi.apply(o12);

        double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];
        o21 = phi.apply(o21);

        DenseMatrix64F output = layer.generateOutput(input);

        assertThat(output.get(0, 0), is(o11));
        assertThat(output.get(0, 1), is(o12));
        assertThat(output.get(1, 0), is(o21));
    }
}
