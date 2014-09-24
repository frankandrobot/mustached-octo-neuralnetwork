package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.ICnnMap;
import org.ejml.data.DenseMatrix64F;

import java.util.LinkedList;
import java.util.List;

class MapInfo
{
    /**
     * could be null
     */
    ICnnMap map;

    DenseMatrix64F yOutput;

    private List<MapInfo> lInputMaps = new LinkedList<MapInfo>();

    private DenseMatrix64F[] aInputs;


    public MapInfo(ICnnMap map)
    {
        if (map != null)
        {
            this.map = map;
            yOutput = new DenseMatrix64F(map.getOutputDim().rows, map.getOutputDim().cols);
        }
    }

    public MapInfo(DenseMatrix64F input)
    {
        yOutput = input;
    }

    public void addInput(MapInfo inputMap)
    {
        lInputMaps.add(inputMap);

        aInputs = new DenseMatrix64F[lInputMaps.size()];

        int len = 0;
        for(MapInfo map:lInputMaps)
        {
            aInputs[len++] = map.yOutput;
        }
    }

    /**
     * Valid only if has map
     */
    public void generateOutput()
    {
        if (aInputs != null)
        {
            this.yOutput = map.generateOutput(aInputs);
        }
    }

    public List<MapInfo> getInputMaps()
    {
        return lInputMaps;
    }
}
