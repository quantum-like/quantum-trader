# VMD batch renderer for the hard-sphere / point-charge trajectories (Materials-Studio CPK look, VMD's bundled Tachyon).
#
#   vmd -dispdev text -e vmd/render_cpk.tcl -args traj/hs_charges_20p20m.lammpstrj frames/vmd_charged [style] [W] [H] [orbit_deg]
#       style = cpk (default) | xray      W H = 1280 720 (default)      orbit_deg = 40 (default, over the whole trajectory)
#   then:  ffmpeg -framerate 30 -i frames/vmd_charged/frame_%05d.tga -c:v libx264 -pix_fmt yuv420p -crf 18 movies/hs_charges_20p20m_vmd.mp4
#   (vmd/render_vmd.sh does both.)  Input: the LAMMPS dump written by tools/export_vmd_travis.py (box in every frame,
#   element column Ar = neutral, Na = +e, Cl = -e, 1 sigma = 4 A).  Needs the pbctools plugin (ships with VMD).
#
if {[llength $argv] < 2} { puts "usage: -args <traj.lammpstrj> <outdir> \[cpk|xray\] \[W\] \[H\] \[orbit_deg\]"; exit 1 }
set traj   [lindex $argv 0]
set outdir [lindex $argv 1]
set style  [expr {[llength $argv] > 2 ? [lindex $argv 2] : "cpk"}]
set W      [expr {[llength $argv] > 3 ? [lindex $argv 3] : 1280}]
set H      [expr {[llength $argv] > 4 ? [lindex $argv 4] : 720}]
set orbit  [expr {[llength $argv] > 5 ? [lindex $argv 5] : 40.0}]
set radius 2.0   ;# sigma/2 in the exported pseudo-Angstrom units (1 sigma = 4 A)
file mkdir $outdir

package require pbctools
set molid [mol new $traj type lammpstrj waitfor all]
set nf [molinfo $molid get numframes]
puts "loaded $traj: $nf frames"

# --- colours: gray neutrals, red +, blue -, light studio background (colour ids 0/1/2 and 8 are redefined)
color change rgb 0 0.15 0.35 0.90    ;# blue  -> Cl (-e)
color change rgb 1 0.85 0.15 0.15    ;# red   -> Na (+e)
color change rgb 2 0.78 0.78 0.80    ;# gray  -> Ar (neutral)
color change rgb 8 0.97 0.97 0.98    ;# white -> background tint
color Display Background white
display backgroundgradient off

# --- materials: soft lighting with a little specular; a ghost material for the x-ray style
material add CPKms
material change ambient   CPKms 0.30
material change diffuse   CPKms 0.75
material change specular  CPKms 0.25
material change shininess CPKms 0.55
material change opacity   CPKms 1.00
material change outline   CPKms 0.00
material add Ghost
material change ambient   Ghost 0.30
material change diffuse   Ghost 0.70
material change specular  Ghost 0.05
material change opacity   Ghost 0.15

# --- one VDW representation per species, sphere radius = sigma/2 exactly
[atomselect $molid all] set radius $radius
mol delrep 0 $molid
foreach {sel colid mat} [list "element Ar" 2 [expr {$style eq "xray" ? "Ghost" : "CPKms"}] "element Na" 1 CPKms "element Cl" 0 CPKms] {
    mol representation VDW 1.0 32.0
    mol color ColorID $colid
    mol selection $sel
    mol material $mat
    mol addrep $molid
}

# --- display: perspective, ambient occlusion + shadows (Tachyon), no axes, faint box
display projection Perspective
display depthcue off
display rendermode Normal
display shadows on
display ambientocclusion on
display aoambient 0.80
display aodirect 0.35
display antialias on
axes location Off
display resize $W $H
pbc box -color gray -width 1.5 -style lines -molid $molid

# --- 3/4 view: reset, tilt down ~22 deg, turn 45 deg, zoom so the orbiting box stays inside the frame
display resetview
rotate x by -22
rotate y by 45
scale by 1.05

# --- Tachyon binary shipped with VMD (e.g. tachyon_MACOSXARM64); override with the TACHYON_BIN environment variable
if {[info exists env(TACHYON_BIN)]} { set tachyon $env(TACHYON_BIN) } else { set tachyon [format "%s/tachyon_%s" $env(VMDDIR) [vmdinfo arch]] }
puts "tachyon: $tachyon"

set dtheta [expr {$nf > 1 ? double($orbit) / ($nf - 1) : 0.0}]
for {set f 0} {$f < $nf} {incr f} {
    animate goto $f
    display update
    set out [format "%s/frame_%05d" $outdir $f]
    render Tachyon $out.dat "\"$tachyon\" -aasamples 4 -res $W $H %s -format TARGA -o %s.tga"
    file delete -force $out.dat
    rotate y by $dtheta
    if {$f % 20 == 0} { puts "frame $f / $nf" }
}
puts "done: $nf frames in $outdir"
exit
