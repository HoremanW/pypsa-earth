rule patch_transmission:
    input:
        buses="resources/" + RDIR + "base_network/all_buses_build_network.geojson",
        lines="resources/" + RDIR + "base_network/all_lines_build_network.geojson",
    output:
        sentinel="resources/" + RDIR + "base_network/.patch_done",
    log:
        "logs/" + RDIR + "patch_transmission.log",
    script:
        "../scripts/patch_transmission.py"
