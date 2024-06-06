from dash_extensions.enrich import Input, Output, callback, clientside_callback, ctx
from dash.exceptions import PreventUpdate


def register_shortcut_callbacks():
    # Switch modebar buttons with keyboard
    clientside_callback(
        """
            function(id) {
                document.addEventListener('keydown', function(event) {
                    if (event.target.nodeName != 'INPUT') {
                        if (event.key == 'a') {
                            document.getElementById('RV-mark-annotations-button').click()
                            event.stopPropogation()
                        }
                        else if (event.key == 'b') {
                            document.getElementById('RV-mark-bad-channels-button').click()
                            event.stopPropogation()
                        }
                        else if (event.key == 'p') {
                            document.getElementById('RV-pan-button').click()
                            event.stopPropogation()
                        }
                        else if (event.key == 's') {
                            document.getElementById('RV-select-segment-button').click()
                            event.stopPropogation()
                        }
                        else if (event.key == 'z') {
                            document.getElementById('RV-zoom-button').click()
                            event.stopPropogation()
                        }
                        else if (event.key == 'Delete' || event.key == 'Backspace') {
                            document.getElementById('RV-delete-annotation-button').click()
                            event.stopPropogation()
                        }
                    }
                });
                return window.dash_clientside.no_update       
            }
        """,
        Output('RV-main-graph', 'id'),
        Input('RV-main-graph', 'id')
    )

    @callback(
        [
            Output('RV-mark-annotations-button', 'className'),
            Output('RV-mark-bad-channels-button', 'className'),
            Output('RV-pan-button', 'className'),
            Output('RV-zoom-button', 'className'),
            Output('RV-select-segment-button', 'className')
        ],
        [
            Input('RV-mark-annotations-button', 'n_clicks'),
            Input('RV-mark-bad-channels-button', 'n_clicks'),
            Input('RV-pan-button', 'n_clicks'),
            Input('RV-zoom-button', 'n_clicks'),
            Input('RV-select-segment-button', 'n_clicks')
        ],
        prevent_initial_call=True
    )
    def highlight_active_button(mark_annotations, mark_bad_channels, pan, zoom, select_segment):
        """Adds 'active-button' and 'inactive-button' class to all buttons that click hidden modebar buttons.
        The active button is highlighted by a blue outline.
        """
        trigger = ctx.triggered_id
        print(f'Active button: {trigger}')

        if 'annotations' in trigger:
            return 'active-button', 'inactive-button', 'inactive-button', 'inactive-button', 'inactive-button'
        elif 'bad-channels' in trigger:
            return 'inactive-button', 'active-button', 'inactive-button', 'inactive-button', 'inactive-button'
        elif 'pan' in trigger:
            return 'inactive-button', 'inactive-button', 'active-button', 'inactive-button', 'inactive-button'
        elif 'zoom' in trigger:
            return 'inactive-button', 'inactive-button', 'inactive-button', 'active-button', 'inactive-button'
        elif 'select' in trigger:
            return 'inactive-button', 'inactive-button', 'inactive-button', 'inactive-button', 'active-button'

        raise PreventUpdate
