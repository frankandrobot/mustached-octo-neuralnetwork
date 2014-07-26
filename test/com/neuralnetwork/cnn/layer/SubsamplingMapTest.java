package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.filter.SimpleSamplingFilter;
import com.neuralnetwork.cnn.layer.builder.FeatureMapBuilder;
import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class SubsamplingMapTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    private final DenseMatrix64F input;
    private final double[] weights;
    private FeatureMapBuilder builder;

    public SubsamplingMapTest()
    {
       input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        weights = new double[]{0.3, 0.3, 0.3, 0.3, 0.4};

        builder = new FeatureMapBuilder()
                .set1DInputSize(4)
                .setNeuron(new MNeuron(phi, weights))
                .setConvolutionFilter(new SimpleSamplingFilter());
    }

    @Test
    public void testSubsampling1()
    {
        SamplingLayer layer = new SamplingLayer(builder);

        layer.generateOutput(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(layer.getOutput().numCols, is(2) );
        assertThat(layer.getOutput().numRows, is(2) );
    }

    @Test
    public void testSubsampling2()
    {
        SamplingLayer layer = new SamplingLayer(builder);

        double o11 = (1 + 2 + 5 + 6);
        o11 = o11 * weights[0] + weights[4];
        o11 = phi.apply(o11);

        double o12 = (3 + 4 + 7 + 8);
        o12 = o12 * weights[0] + weights[4];
        o12 = phi.apply(o12);

        double o21 = (9 + 10 + 13 + 14);
        o21 = o21 * weights[0] + weights[4];
        o21 = phi.apply(o21);

        layer.generateOutput(input);

        assertThat(layer.getOutput().get(0,0), is(o11));
        assertThat(layer.getOutput().get(0,1), is(o12));
        assertThat(layer.getOutput().get(1,0), is(o21));
    }

//    @Test
//    public void testRawOutputFuncs()
//    {
//        SubSamplingLayer layer = new SubSamplingLayer(builder);
//
//        DenseMatrix64F output = layer.generateOutput(input);
//
//        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
//        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
//        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
//        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));
//
//        final double o11 = (1 + 2 + 5 + 6) * weights[0] + weights[1];
//
//        assertThat(featureMap.rawoutput(input, 0, 0), is(o11));
//
//        final double o12 = (3 + 4 + 7 + 8) * weights[0] + weights[1];
//
//        assertThat(featureMap.rawoutput(input,0,1), is(o12));
//
//        final double o21 = (9 + 10 + 13 + 14) * weights[0] + weights[1];
//
//        assertThat(featureMap.rawoutput(input,1,0), is(o21));
//    }

//    @Test
//    public void testMapToFeatureMapRow()
//    {
//        final double[] weights = {0.3, 0.4};
//
//        FeatureMap.Builder builder = new FeatureMap.Builder();
//        builder.setNeuron(new MNeuron(phi, weights));
//        builder.setReceptiveFieldSize(4 * 4);
//        builder.set1DInputSize(8);
//
//        FeatureMap featureMap = new SubSamplingMapOld(builder);
//
//        for(int inputRow=0; inputRow<4; inputRow++)
//        {
//            for(int weight=0; weight<4*4; weight++)
//            {
//                int row = featureMap.featureMapRowPosition(weight, inputRow);
//                assertThat(row, is(0));
//            }
//        }
//
//        for(int inputRow=4; inputRow<8; inputRow++)
//        {
//            for(int weight=4; weight<4*4; weight++)
//            {
//                //System.out.println("inputRow:"+inputRow+" weight:"+weight);
//                int row = featureMap.featureMapRowPosition(weight, inputRow);
//                assertThat(row, is(1));
//            }
//        }
//    }

//    @Test
//    public void testMapToFeatureMapCol()
//    {
//        final double[] weights = {0.3, 0.4};
//
//        FeatureMap.Builder builder = new FeatureMap.Builder();
//        builder.setNeuron(new MNeuron(phi, weights));
//        builder.setReceptiveFieldSize(4 * 4);
//        builder.set1DInputSize(8);
//
//        FeatureMap featureMap = new SubSamplingMapOld(builder);
//
//        for(int inputCol =0; inputCol <4; inputCol++)
//        {
//            for(int weight=0; weight<4*4; weight++)
//            {
//                int col = featureMap.featureMapColPosition(weight, inputCol);
//                assertThat(col, is(0));
//            }
//        }
//
//        for(int inputCol =4; inputCol <8; inputCol++)
//        {
//            for(int weight=4; weight<4*4; weight++)
//            {
//                //System.out.println("inputRow:"+inputRow+" weight:"+weight);
//                int col = featureMap.featureMapColPosition(weight, inputCol);
//                assertThat(col, is(1));
//            }
//        }
//    }
}
