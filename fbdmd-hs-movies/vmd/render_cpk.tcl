# VMD batch renderer for the hard-sphere / point-charge trajectories (Materials-Studio CPK look, VMD's bundled Tachyon).
#
#   vmd -dispdev text -e vmd/render_cpk.tcl -args traj/hs_charges_20p20m.lammpstrj frames/vmd_charged [style] [W] [H] [orbit_deg] [aa]
#       style = cpk (default) | xray      W H = 1280 720      orbit_deg = 40 over the whole trajectory      aa = Tachyon AA samples (8)
#   then:  ffmpeg -framerate 30 -i frames/vmd_charged/frame_%05d.tga -c:v libx264 -pix_fmt yuv420p -crf 18 movies/<name>.mp4
#   (vmd/render_vmd.sh does both.)  Input: the LAMMPS dump written by tools/export_vmd_travis.py (box in every frame,
#   element column Ar = neutral, Na = +e, Cl = -e, 1 sigma = 4 A).  Needs the pbctools plugin (ships with VMD).
#   Tachyon binary: $TACHYON_BIN (VMD's startup.command exports it), else $VMDDIR/tachyon_<arch>, else 'tachyon' on PATH.
#   macOS: if the Tachyon child is killed by Gatekeeper, run  xattr -dr com.apple.quarantine /Applications/VMD*.app
#
if {[llength $argv] < 2} { puts "usage: -args <traj.lammpstrj> <outdir> \[cpk|xray\] \[W\] \[H\] \[orbit_deg\] \[aa\]"; exit 1 }
set traj   [lindex $argv 0]
set outdir [lindex $argv 1]
set style  [expr {[llength $argv] > 2 ? [lindex $argv 2] : "cpk"}]
set W      [expr {[llength $argv] > 3 ? [lindex $argv 3] : 1280}]
set H      [expr {[llength $argv] > 4 ? [lindex $argv 4] : 720}]
set orbit  [expr {[llength $argv] > 5 ? [lindex $argv 5] : 40.0}]
set aa     [expr {[llength $argv] > 6 ? [lindex $argv 6] : 8}]
set radius 2.0   ;# sigma/2 in the exported pseudo-Angstrom units (1 sigma = 4 A)
set tilt   22.0  ;# camera elevation in degrees (positive = looking down onto the box; verify the sign once in view_cpk.tcl)
file mkdir $outdir

# --- Tachyon binary
if {[info exists env(TACHYON_BIN)]} {
    set tachyon $env(TACHYON_BIN)
} elseif {[info exists env(VMDDIR)]} {
    set tachyon [format "%s/tachyon_%s" $env(VMDDIR) [vmdinfo arch]]
} else {
    set tachyon tachyon
}
if {![file executable $tachyon] && [auto_execok $tachyon] eq ""} {
    puts "ERROR: Tachyon binary not found ($tachyon). Set TACHYON_BIN=/Applications/VMD.app/Contents/vmd/tachyon_MACOSXARM64 (or similar)."
    exit 1
}
puts "tachyon: $tachyon"

package require pbctools
set molid [mol new $traj type lammpstrj waitfor all]
set nf [molinfo $molid get numframes]
puts "loaded $traj: $nf frames"
# make the element field reliable whatever the plugin did with the 'element' column (types 1/2/3 = Ar/Na/Cl)
foreach {t e} {1 Ar 2 Na 3 Cl} { set s [atomselect $molid "type $t"]; $s set element $e; $s set name $e; $s delete }

# --- colours: gray neutrals, red +, blue -, light studio background, dark-gray box (colour ids 0/1/2/8/16 redefined)
color change rgb 0  0.15 0.35 0.90    ;# blue  -> Cl (-e)
color change rgb 1  0.85 0.15 0.15    ;# red   -> Na (+e)
color change rgb 2  0.78 0.78 0.80    ;# gray  -> Ar (neutral)
color change rgb 8  0.97 0.97 0.98    ;# white -> background tint
color change rgb 16 0.40 0.40 0.42    ;# black -> box edges (dark gray)
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

# --- one VDW representation per species, sphere radius = sigma/2 exactly (VDW scale 1.0 x per-atom radius)
set all [atomselect $molid all]; $all set radius $radius; $all delete
while {[molinfo $molid get numreps] > 0} { mol delrep 0 $molid }
foreach {sel colid mat} [list "element Ar" 2 [expr {$style eq "xray" ? "Ghost" : "CPKms"}] "element Na" 1 CPKms "element Cl" 0 CPKms] {
    mol representation VDW 1.0 32.0
    mol color ColorID $colid
    mol selection $sel
    mol material $mat
    mol addrep $molid
}

# --- display: perspective, ambient occlusion + shadows (written into the Tachyon scene), no axes, faint box
display projection Perspective
display depthcue off
display rendermode Normal
display shadows on
display ambientocclusion on
display aoambient 0.80
display aodirect 0.35
axes location Off
catch {display resize $W $H}
catch {render imagesize Tachyon $W $H}
pbc box -color black -width 1.5 -style lines -molid $molid

# --- 3/4 view: turn 45 deg about the vertical, then tilt the camera; zoom so the orbiting box stays inside the frame
display resetview
rotate y by 45
rotate x by $tilt
scale by 1.05

set trans [expr {$style eq "xray" ? " -trans_max_surfaces 24" : ""}]
set dtheta [expr {$nf > 1 ? double($orbit) / ($nf - 1) : 0.0}]
for {set f 0} {$f < $nf} {incr f} {
    animate goto $f
    display update
    set out [format "%s/frame_%05d" $outdir $f]
    # exactly ONE %s (VMD replaces every %s with the scene file name)
    render Tachyon $out.dat "\"$tachyon\" -aasamples $aa -res $W $H$trans %s -format TARGA -o $out.tga"
    file delete -force $out.dat
    # orbit about the box's vertical axis: undo the tilt, spin, re-tilt (rotate acts about screen axes)
    rotate x by [expr {-$tilt}]
    rotate y by $dtheta
    rotate x by $tilt
    if {$f % 20 == 0} { puts "frame $f / $nf" }
}
puts "done: $nf frames in $outdir"
quit
