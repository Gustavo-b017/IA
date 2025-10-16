from 11_combined_features_by_z_plus_surface import main
if __name__=="__main__":
    # Zoom tighter by lowering pad and keep camera default
    main(out="combined_features_by_z_plus_surface_zoomed.gif", pad=0.3, elev=30, azim=-60)
