package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.filter.SimpleSamplingFilter;
import com.neuralnetwork.cnn.layer.builder.SamplingLayerBuilder;
import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class SamplingLayerTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    private DenseMatrix64F input;
    private double[] weights;
    private SamplingLayerBuilder builder;

    @Before
    public void setup()
    {
       input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        weights = new double[]{0.4, 0.3, 0.3, 0.3, 0.3};

        builder = new SamplingLayerBuilder()
                .set1DInputSize(4)
                .setNeuron(new Neuron(phi, weights))
                .setFilter(new SimpleSamplingFilter());
    }

    @Test
    public void testSubsampling1()
    {
        SamplingLayer layer = new SamplingLayer(builder);

        DenseMatrix64F output = layer.generateOutput(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(output.numCols, is(2) );
        assertThat(output.numRows, is(2) );
    }

    @Test
    public void testSubsampling2()
    {
        SamplingLayer layer = new SamplingLayer(builder);

        double o11 = (1 + 2 + 5 + 6);
        o11 = o11 * weights[1] + weights[0];
        o11 = phi.apply(o11);

        double o12 = (3 + 4 + 7 + 8);
        o12 = o12 * weights[1] + weights[0];
        o12 = phi.apply(o12);

        double o21 = (9 + 10 + 13 + 14);
        o21 = o21 * weights[1] + weights[0];
        o21 = phi.apply(o21);

        DenseMatrix64F output = layer.generateOutput(input);

        assertThat(output.get(0,0), is(o11));
        assertThat(output.get(0,1), is(o12));
        assertThat(output.get(1,0), is(o21));
    }
}
